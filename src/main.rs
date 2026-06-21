use std::{
    collections::{HashMap, HashSet},
    env,
    ffi::OsStr,
    net::SocketAddr,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex as StdMutex,
    },
    time::{Duration, Instant, SystemTime},
};

use anyhow::{anyhow, Context};
use axum::{
    body::Body,
    extract::{Request, State},
    http::{
        header::{
            ACCEPT, ACCEPT_ENCODING, CACHE_CONTROL, CONTENT_LENGTH, CONTENT_TYPE,
            IF_MODIFIED_SINCE, IF_NONE_MATCH, IF_RANGE, RANGE, USER_AGENT, VARY,
        },
        HeaderMap, HeaderName, HeaderValue, Method, Response, StatusCode, Uri,
    },
    routing::any,
    Router,
};
use bytes::Bytes;
use futures_util::StreamExt;
use image::{GenericImageView, ImageFormat};
use reqwest::Client;
use rgb::AsPixels;
use sha2::{Digest, Sha256};
use tokio::{
    fs,
    io::AsyncWriteExt,
    net::TcpListener,
    sync::{Mutex as AsyncMutex, OwnedMutexGuard, Semaphore},
};
use tokio_util::io::ReaderStream;
use tower_http::trace::TraceLayer;
use tracing::{debug, error, info, warn};
use tracing_subscriber::{fmt, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};
use url::{form_urlencoded, Url};

const DEFAULT_LISTEN_ADDR: &str = "127.0.0.1:9000";
const DEFAULT_UPSTREAM_BASE: &str = "http://127.0.0.1:9100";
const DEFAULT_CACHE_DIR: &str = r"C:\minio_img_cache";
const DEFAULT_LOG_DIR: &str = ".\\logs";
const DEFAULT_MAX_TRANSFORM_BYTES: u64 = 25 * 1024 * 1024;
const DEFAULT_MAX_PIXELS: u64 = 40_000_000;
const DEFAULT_AVIF_QUALITY: f32 = 75.0;
const DEFAULT_AVIF_SPEED: u8 = 6;
const DEFAULT_WEBP_QUALITY: f32 = 82.0;
const DEFAULT_CACHE_VERSION: &str = "1";
const DEFAULT_UPSTREAM_CONNECT_TIMEOUT_SECS: u64 = 3;
const DEFAULT_UPSTREAM_REQUEST_TIMEOUT_SECS: u64 = 30;
const DEFAULT_CACHE_CLEANUP_ENABLED: bool = true;
const DEFAULT_CACHE_MAX_BYTES: u64 = 10 * 1024 * 1024 * 1024;
const DEFAULT_CACHE_MAX_AGE_DAYS: u64 = 90;
const DEFAULT_CACHE_CLEANUP_INTERVAL_SECS: u64 = 3600;

#[derive(Clone)]
struct AppState {
    cfg: Arc<Config>,
    client: Client,
    encode_limiter: Arc<Semaphore>,
    in_flight_encodes: Arc<StdMutex<HashMap<String, Arc<AsyncMutex<()>>>>>,
    temp_file_counter: Arc<AtomicU64>,
}

#[derive(Debug, Clone)]
struct Config {
    listen_addr: SocketAddr,
    upstream_base: Url,
    cache_dir: PathBuf,
    max_transform_bytes: u64,
    max_pixels: u64,
    avif_quality: f32,
    avif_speed: u8,
    webp_quality: f32,
    cache_version: String,
    enable_accept_negotiation: bool,
    max_concurrent_encodes: usize,
    upstream_connect_timeout: Duration,
    upstream_request_timeout: Duration,
    cache_cleanup_enabled: bool,
    cache_max_bytes: u64,
    cache_max_age: Duration,
    cache_cleanup_interval: Duration,
}

#[derive(Debug)]
struct CacheFile {
    path: PathBuf,
    len: u64,
    modified: SystemTime,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TargetFormat {
    Avif,
    Webp,
}

impl TargetFormat {
    fn extension(self) -> &'static str {
        match self {
            TargetFormat::Avif => "avif",
            TargetFormat::Webp => "webp",
        }
    }

    fn content_type(self) -> &'static str {
        match self {
            TargetFormat::Avif => "image/avif",
            TargetFormat::Webp => "image/webp",
        }
    }
}

#[derive(Debug, Clone)]
struct TransformDecision {
    target: TargetFormat,
    upstream_query: Option<String>,
}

#[derive(Debug, Clone)]
struct CacheEntry {
    key: String,
    final_path: PathBuf,
}

struct EncodeFlightGuard {
    key: String,
    flights: Arc<StdMutex<HashMap<String, Arc<AsyncMutex<()>>>>>,
    lock: Arc<AsyncMutex<()>>,
    _guard: OwnedMutexGuard<()>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let _log_guard = init_logging()?;
    let cfg = Arc::new(Config::from_env()?);

    fs::create_dir_all(&cfg.cache_dir)
        .await
        .with_context(|| format!("failed to create cache dir {}", cfg.cache_dir.display()))?;

    let client = Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .connect_timeout(cfg.upstream_connect_timeout)
        .timeout(cfg.upstream_request_timeout)
        .pool_idle_timeout(Duration::from_secs(90))
        .build()
        .context("failed to build reqwest client")?;

    let state = AppState {
        encode_limiter: Arc::new(Semaphore::new(cfg.max_concurrent_encodes)),
        in_flight_encodes: Arc::new(StdMutex::new(HashMap::new())),
        temp_file_counter: Arc::new(AtomicU64::new(0)),
        cfg: cfg.clone(),
        client,
    };

    if cfg.cache_cleanup_enabled {
        tokio::spawn(cache_cleanup_loop(cfg.clone()));
    }

    let app = build_router(state);

    let listener = TcpListener::bind(cfg.listen_addr)
        .await
        .with_context(|| format!("failed to bind {}", cfg.listen_addr))?;

    info!(
        listen_addr = %cfg.listen_addr,
        upstream = %cfg.upstream_base,
        cache_dir = %cfg.cache_dir.display(),
        max_transform_bytes = cfg.max_transform_bytes,
        max_concurrent_encodes = cfg.max_concurrent_encodes,
        accept_negotiation = cfg.enable_accept_negotiation,
        cache_cleanup_enabled = cfg.cache_cleanup_enabled,
        "image optimizer started"
    );

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .context("server failed")?;

    info!("image optimizer stopped");
    Ok(())
}

fn build_router(state: AppState) -> Router {
    Router::new()
        .fallback(any(proxy_handler))
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

async fn shutdown_signal() {
    if let Err(err) = tokio::signal::ctrl_c().await {
        error!(error = %err, "failed to install Ctrl+C handler");
        return;
    }
    info!("shutdown signal received");
}

fn init_logging() -> anyhow::Result<tracing_appender::non_blocking::WorkerGuard> {
    let log_dir = env::var("LOG_DIR").unwrap_or_else(|_| DEFAULT_LOG_DIR.to_string());
    std::fs::create_dir_all(&log_dir)
        .with_context(|| format!("failed to create log dir {}", log_dir))?;

    let file_appender = tracing_appender::rolling::daily(&log_dir, "image_optimizer.log");
    let (file_writer, guard) = tracing_appender::non_blocking(file_appender);

    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("image_optimizer=info,tower_http=info"));

    tracing_subscriber::registry()
        .with(filter)
        .with(fmt::layer().with_writer(std::io::stdout).with_ansi(false))
        .with(fmt::layer().with_writer(file_writer).with_ansi(false))
        .init();

    Ok(guard)
}

async fn proxy_handler(State(state): State<AppState>, req: Request) -> Response<Body> {
    let started = Instant::now();
    let method = req.method().clone();
    let uri = req.uri().clone();
    let headers = req.headers().clone();

    match method {
        Method::GET => handle_get(state, uri, headers, started).await,
        Method::HEAD => {
            let upstream_uri = uri_with_query(&uri, strip_format_query_raw(uri.query()).as_deref());
            proxy_upstream(state, &Method::HEAD, &upstream_uri, &headers).await
        }
        _ => response_with_status(StatusCode::METHOD_NOT_ALLOWED, "method not allowed"),
    }
}

async fn handle_get(
    state: AppState,
    uri: Uri,
    headers: HeaderMap,
    started: Instant,
) -> Response<Body> {
    let Some(decision) = decide_transform(&uri, &headers, &state.cfg) else {
        let upstream_uri = uri_with_query(&uri, strip_format_query_raw(uri.query()).as_deref());
        return proxy_upstream(state, &Method::GET, &upstream_uri, &headers).await;
    };

    let cache_entry = cache_entry(&state.cfg, &uri, &decision);
    if let Ok(file) = fs::File::open(&cache_entry.final_path).await {
        info!(
            path = %uri,
            format = decision.target.extension(),
            elapsed_ms = started.elapsed().as_millis(),
            "[Cache Hit]"
        );
        return cached_file_response(file, decision.target).await;
    }

    info!(
        path = %uri,
        format = decision.target.extension(),
        "[Cache Miss]"
    );

    let flight_guard = acquire_encode_flight(&state, cache_entry.key.clone()).await;
    if let Ok(file) = fs::File::open(&cache_entry.final_path).await {
        info!(
            path = %uri,
            format = decision.target.extension(),
            elapsed_ms = started.elapsed().as_millis(),
            "[Cache Hit After Wait]"
        );
        return cached_file_response(file, decision.target).await;
    }

    let upstream_uri = uri_with_query(&uri, decision.upstream_query.as_deref());
    let upstream_response =
        match fetch_upstream(&state, &Method::GET, &upstream_uri, &headers).await {
            Ok(response) => response,
            Err(err) => {
                error!(path = %uri, error = %err, "upstream request failed");
                return response_with_status(StatusCode::BAD_GATEWAY, "bad gateway");
            }
        };

    if !upstream_response.status().is_success() {
        return reqwest_response_to_axum(upstream_response, false).await;
    }

    if should_skip_for_size(upstream_response.headers(), state.cfg.max_transform_bytes) {
        info!(
            path = %uri,
            max_transform_bytes = state.cfg.max_transform_bytes,
            "source image too large for transform; streaming original"
        );
        return reqwest_response_to_axum(upstream_response, true).await;
    }

    let source_headers = upstream_response.headers().clone();
    let source_bytes = match read_limited(upstream_response, state.cfg.max_transform_bytes).await {
        Ok(bytes) => bytes,
        Err(ReadBodyError::TooLarge) => {
            warn!(
                path = %uri,
                max_transform_bytes = state.cfg.max_transform_bytes,
                "source image exceeded transform limit while reading; refetching original"
            );
            return proxy_upstream(state, &Method::GET, &upstream_uri, &headers).await;
        }
        Err(ReadBodyError::Request(err)) => {
            error!(path = %uri, error = %err, "failed to read upstream body");
            return response_with_status(StatusCode::BAD_GATEWAY, "bad gateway");
        }
    };

    let permit_started = Instant::now();
    let permit = match state.encode_limiter.clone().acquire_owned().await {
        Ok(permit) => permit,
        Err(_) => {
            return response_with_status(StatusCode::SERVICE_UNAVAILABLE, "service unavailable")
        }
    };
    let wait_ms = permit_started.elapsed().as_millis();
    if wait_ms > 0 {
        debug!(path = %uri, wait_ms, "waited for encode permit");
    }

    let cfg = state.cfg.clone();
    let target = decision.target;
    let source_for_encode = source_bytes.clone();
    let encode_started = Instant::now();
    let converted = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        encode_image(&source_for_encode, target, &cfg)
    })
    .await;

    let converted = match converted {
        Ok(Ok(bytes)) => bytes,
        Ok(Err(err)) => {
            warn!(
                path = %uri,
                format = decision.target.extension(),
                error = %err,
                "[Convert Failed]; returning original"
            );
            return original_bytes_response(source_headers, source_bytes);
        }
        Err(err) => {
            error!(path = %uri, error = %err, "encoder task failed");
            return original_bytes_response(source_headers, source_bytes);
        }
    };

    info!(
        path = %uri,
        format = decision.target.extension(),
        encode_ms = encode_started.elapsed().as_millis(),
        elapsed_ms = started.elapsed().as_millis(),
        "[Format Converted]"
    );

    if let Err(err) = write_cache_file(&state, &cache_entry, &converted).await {
        warn!(
            path = %uri,
            cache_path = %cache_entry.final_path.display(),
            error = %err,
            "[Cache Write Failed]"
        );
    }

    drop(flight_guard);

    converted_response(converted, decision.target)
}

fn response_with_status(status: StatusCode, body: &'static str) -> Response<Body> {
    Response::builder()
        .status(status)
        .body(Body::from(body))
        .expect("static response is valid")
}

async fn acquire_encode_flight(state: &AppState, key: String) -> EncodeFlightGuard {
    let lock = {
        let mut flights = state
            .in_flight_encodes
            .lock()
            .expect("in-flight encode lock table poisoned");
        flights
            .entry(key.clone())
            .or_insert_with(|| Arc::new(AsyncMutex::new(())))
            .clone()
    };
    let guard = lock.clone().lock_owned().await;
    EncodeFlightGuard {
        key,
        flights: state.in_flight_encodes.clone(),
        lock,
        _guard: guard,
    }
}

impl Drop for EncodeFlightGuard {
    fn drop(&mut self) {
        if Arc::strong_count(&self.lock) <= 3 {
            if let Ok(mut flights) = self.flights.lock() {
                if flights
                    .get(&self.key)
                    .is_some_and(|current| Arc::ptr_eq(current, &self.lock))
                {
                    flights.remove(&self.key);
                }
            }
        }
    }
}

async fn proxy_upstream(
    state: AppState,
    method: &Method,
    uri: &Uri,
    headers: &HeaderMap,
) -> Response<Body> {
    match fetch_upstream(&state, method, uri, headers).await {
        Ok(response) => reqwest_response_to_axum(response, true).await,
        Err(err) => {
            error!(path = %uri, error = %err, "upstream request failed");
            response_with_status(StatusCode::BAD_GATEWAY, "bad gateway")
        }
    }
}

async fn fetch_upstream(
    state: &AppState,
    method: &Method,
    uri: &Uri,
    headers: &HeaderMap,
) -> anyhow::Result<reqwest::Response> {
    let upstream_url = build_upstream_url(&state.cfg.upstream_base, uri)?;
    let reqwest_method = reqwest::Method::from_bytes(method.as_str().as_bytes())
        .context("failed to convert HTTP method")?;
    let mut request = state.client.request(reqwest_method, upstream_url);

    for name in forwarded_request_headers() {
        if let Some(value) = headers.get(name) {
            request = request.header(name.as_str(), value);
        }
    }

    request
        .send()
        .await
        .context("failed to send upstream request")
}

async fn reqwest_response_to_axum(response: reqwest::Response, streaming: bool) -> Response<Body> {
    let status = response.status();
    let headers = response.headers().clone();
    let mut builder = Response::builder().status(status);
    copy_response_headers(builder.headers_mut().unwrap(), &headers, streaming);

    if streaming {
        let stream = response
            .bytes_stream()
            .map(|result| result.map_err(std::io::Error::other));
        builder
            .body(Body::from_stream(stream))
            .expect("stream response is valid")
    } else {
        match response.bytes().await {
            Ok(bytes) => builder
                .body(Body::from(bytes))
                .expect("bytes response is valid"),
            Err(err) => {
                error!(error = %err, "failed to read upstream response");
                response_with_status(StatusCode::BAD_GATEWAY, "bad gateway")
            }
        }
    }
}

fn copy_response_headers(target: &mut HeaderMap, source: &HeaderMap, include_length: bool) {
    static HOP_BY_HOP: &[&str] = &[
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    ];
    let hop_by_hop: HashSet<&str> = HOP_BY_HOP.iter().copied().collect();

    for (name, value) in source {
        if hop_by_hop.contains(name.as_str()) {
            continue;
        }
        if !include_length && name == CONTENT_LENGTH {
            continue;
        }
        target.insert(name.clone(), value.clone());
    }
}

fn forwarded_request_headers() -> &'static [HeaderName] {
    static HEADERS: std::sync::OnceLock<Vec<HeaderName>> = std::sync::OnceLock::new();
    HEADERS.get_or_init(|| {
        vec![
            ACCEPT,
            ACCEPT_ENCODING,
            IF_NONE_MATCH,
            IF_MODIFIED_SINCE,
            RANGE,
            IF_RANGE,
            USER_AGENT,
        ]
    })
}

fn decide_transform(uri: &Uri, headers: &HeaderMap, cfg: &Config) -> Option<TransformDecision> {
    if headers.contains_key(RANGE) {
        return None;
    }
    if !is_transformable_image_path(uri.path()) {
        return None;
    }

    let explicit_format = parse_query(uri.query())
        .iter()
        .rev()
        .find(|(key, _)| key.eq_ignore_ascii_case("format"))
        .and_then(|(_, value)| parse_target_format(value));

    let target = match explicit_format {
        Some(target) => Some(target),
        None if cfg.enable_accept_negotiation => negotiate_accept(headers.get(ACCEPT)),
        None => None,
    }?;

    Some(TransformDecision {
        target,
        upstream_query: strip_format_query_raw(uri.query()),
    })
}

fn parse_query(query: Option<&str>) -> Vec<(String, String)> {
    query
        .map(|raw| {
            form_urlencoded::parse(raw.as_bytes())
                .map(|(key, value)| (key.into_owned(), value.into_owned()))
                .collect()
        })
        .unwrap_or_default()
}

fn strip_format_query_raw(query: Option<&str>) -> Option<String> {
    let query = query?;
    let kept: Vec<&str> = query
        .split('&')
        .filter(|pair| !query_pair_key_eq_ignore_ascii_case(pair, "format"))
        .collect();
    (!kept.is_empty()).then(|| kept.join("&"))
}

fn query_pair_key_eq_ignore_ascii_case(pair: &str, expected: &str) -> bool {
    let key = pair.split_once('=').map(|(key, _)| key).unwrap_or(pair);
    let parse_input = format!("{key}=");
    form_urlencoded::parse(parse_input.as_bytes())
        .next()
        .is_some_and(|(decoded_key, _)| decoded_key.eq_ignore_ascii_case(expected))
}

fn parse_target_format(value: &str) -> Option<TargetFormat> {
    match value.trim().to_ascii_lowercase().as_str() {
        "avif" => Some(TargetFormat::Avif),
        "webp" => Some(TargetFormat::Webp),
        _ => None,
    }
}

fn negotiate_accept(value: Option<&HeaderValue>) -> Option<TargetFormat> {
    let accept = value.and_then(|value| value.to_str().ok())?;
    if accepts_mime(accept, "image/avif") {
        return Some(TargetFormat::Avif);
    }
    if accepts_mime(accept, "image/webp") {
        return Some(TargetFormat::Webp);
    }
    None
}

fn accepts_mime(accept: &str, wanted: &str) -> bool {
    accept.split(',').any(|part| {
        let mut segments = part.split(';').map(str::trim);
        let Some(mime) = segments.next() else {
            return false;
        };
        if !mime.eq_ignore_ascii_case(wanted) {
            return false;
        }
        let q = segments
            .find_map(|segment| {
                let (key, value) = segment.split_once('=')?;
                key.trim()
                    .eq_ignore_ascii_case("q")
                    .then(|| value.trim().parse::<f32>().ok())
                    .flatten()
            })
            .unwrap_or(1.0);
        q > 0.0
    })
}

fn is_transformable_image_path(path: &str) -> bool {
    let path_without_trailing_slash = path.trim_end_matches('/');
    let Some(ext) = Path::new(path_without_trailing_slash)
        .extension()
        .and_then(OsStr::to_str)
    else {
        return false;
    };
    matches!(ext.to_ascii_lowercase().as_str(), "jpg" | "jpeg" | "png")
}

fn cache_entry(cfg: &Config, uri: &Uri, decision: &TransformDecision) -> CacheEntry {
    let mut hasher = Sha256::new();
    hasher.update(b"GET\n");
    hasher.update(uri.path().as_bytes());
    hasher.update(b"\n");
    if let Some(query) = decision.upstream_query.as_deref() {
        hasher.update(query.as_bytes());
    }
    hasher.update(b"\n");
    hasher.update(decision.target.extension().as_bytes());
    hasher.update(b"\n");
    hasher.update(encoder_fingerprint(cfg, decision.target).as_bytes());
    hasher.update(b"\n");
    hasher.update(cfg.cache_version.as_bytes());

    let digest = hex::encode(hasher.finalize());
    let shard = &digest[..2];
    let final_path = cfg
        .cache_dir
        .join(decision.target.extension())
        .join(shard)
        .join(format!("{}.{}", digest, decision.target.extension()));
    CacheEntry {
        key: digest,
        final_path,
    }
}

fn encoder_fingerprint(cfg: &Config, target: TargetFormat) -> String {
    match target {
        TargetFormat::Avif => format!("avif:q={}:speed={}", cfg.avif_quality, cfg.avif_speed),
        TargetFormat::Webp => format!("webp:q={}", cfg.webp_quality),
    }
}

fn uri_with_query(uri: &Uri, query: Option<&str>) -> Uri {
    let mut parts = uri.clone().into_parts();
    let path = uri.path();
    let path_and_query = match query {
        Some(query) if !query.is_empty() => format!("{}?{}", path, query),
        _ => path.to_string(),
    };
    parts.path_and_query = Some(
        path_and_query
            .parse()
            .expect("path and query generated from valid URI parts"),
    );
    Uri::from_parts(parts).expect("URI generated from valid URI parts")
}

fn build_upstream_url(base: &Url, uri: &Uri) -> anyhow::Result<Url> {
    let mut upstream = base.as_str().trim_end_matches('/').to_string();
    upstream.push_str(uri.path());
    if let Some(query) = uri.query() {
        upstream.push('?');
        upstream.push_str(query);
    }
    Url::parse(&upstream).context("failed to build upstream URL")
}

fn should_skip_for_size(headers: &HeaderMap, max_bytes: u64) -> bool {
    headers
        .get(CONTENT_LENGTH)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<u64>().ok())
        .is_some_and(|len| len > max_bytes)
}

#[derive(Debug)]
enum ReadBodyError {
    TooLarge,
    Request(reqwest::Error),
}

async fn read_limited(response: reqwest::Response, max_bytes: u64) -> Result<Bytes, ReadBodyError> {
    let mut stream = response.bytes_stream();
    let mut body = bytes::BytesMut::new();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(ReadBodyError::Request)?;
        if (body.len() as u64) + (chunk.len() as u64) > max_bytes {
            return Err(ReadBodyError::TooLarge);
        }
        body.extend_from_slice(&chunk);
    }
    Ok(body.freeze())
}

fn encode_image(source: &[u8], target: TargetFormat, cfg: &Config) -> anyhow::Result<Bytes> {
    let format = image::guess_format(source).context("failed to detect image format")?;
    if !matches!(format, ImageFormat::Jpeg | ImageFormat::Png) {
        return Err(anyhow!("unsupported source image format: {:?}", format));
    }

    let image = image::load_from_memory_with_format(source, format)
        .context("failed to decode source image")?;
    let (width, height) = image.dimensions();
    ensure_pixel_limit(width, height, cfg.max_pixels)?;
    let rgba_image = image.to_rgba8();
    let rgba = rgba_image.into_raw();

    match target {
        TargetFormat::Avif => encode_avif(&rgba, width, height, cfg),
        TargetFormat::Webp => encode_webp(&rgba, width, height, cfg),
    }
}

fn ensure_pixel_limit(width: u32, height: u32, max_pixels: u64) -> anyhow::Result<()> {
    let pixels = u64::from(width).saturating_mul(u64::from(height));
    if pixels > max_pixels {
        return Err(anyhow!(
            "image dimensions exceed MAX_PIXELS: {}x{}={} > {}",
            width,
            height,
            pixels,
            max_pixels
        ));
    }
    Ok(())
}

fn encode_avif(rgba: &[u8], width: u32, height: u32, cfg: &Config) -> anyhow::Result<Bytes> {
    let img: &[rgb::RGBA8] = rgba.as_pixels();
    let encoder = ravif::Encoder::new()
        .with_quality(cfg.avif_quality)
        .with_speed(cfg.avif_speed);
    let encoded = encoder
        .encode_rgba(ravif::Img::new(img, width as usize, height as usize))
        .context("failed to encode AVIF")?;
    Ok(Bytes::from(encoded.avif_file))
}

fn encode_webp(rgba: &[u8], width: u32, height: u32, cfg: &Config) -> anyhow::Result<Bytes> {
    let encoder = webp::Encoder::from_rgba(rgba, width, height);
    let encoded = encoder.encode(cfg.webp_quality);
    Ok(Bytes::copy_from_slice(&encoded))
}

async fn write_cache_file(
    state: &AppState,
    entry: &CacheEntry,
    bytes: &[u8],
) -> anyhow::Result<()> {
    let parent = entry
        .final_path
        .parent()
        .ok_or_else(|| anyhow!("cache path has no parent"))?;
    fs::create_dir_all(parent).await?;

    let temp_path = cache_temp_path(state, entry);
    let mut file = fs::File::create(&temp_path).await?;
    file.write_all(bytes).await?;
    file.flush().await?;
    drop(file);

    match fs::rename(&temp_path, &entry.final_path).await {
        Ok(()) => Ok(()),
        Err(err) => {
            let _ = fs::remove_file(&temp_path).await;
            Err(err.into())
        }
    }
}

fn cache_temp_path(state: &AppState, entry: &CacheEntry) -> PathBuf {
    let extension = entry
        .final_path
        .extension()
        .and_then(OsStr::to_str)
        .unwrap_or("cache");
    let sequence = state.temp_file_counter.fetch_add(1, Ordering::Relaxed);
    entry
        .final_path
        .with_extension(format!("{extension}.tmp-{}-{sequence}", std::process::id()))
}

async fn cached_file_response(file: fs::File, target: TargetFormat) -> Response<Body> {
    let len = file.metadata().await.ok().map(|metadata| metadata.len());
    let stream = ReaderStream::new(file);
    let mut builder = Response::builder()
        .status(StatusCode::OK)
        .header(CONTENT_TYPE, target.content_type())
        .header(CACHE_CONTROL, "public, max-age=31536000, immutable")
        .header(VARY, "Accept");
    if let Some(len) = len {
        builder = builder.header(CONTENT_LENGTH, len);
    }
    builder
        .body(Body::from_stream(stream))
        .expect("cached file response is valid")
}

fn converted_response(bytes: Bytes, target: TargetFormat) -> Response<Body> {
    Response::builder()
        .status(StatusCode::OK)
        .header(CONTENT_TYPE, target.content_type())
        .header(CONTENT_LENGTH, bytes.len())
        .header(CACHE_CONTROL, "public, max-age=31536000, immutable")
        .header(VARY, "Accept")
        .body(Body::from(bytes))
        .expect("converted response is valid")
}

fn original_bytes_response(headers: HeaderMap, bytes: Bytes) -> Response<Body> {
    let mut builder = Response::builder().status(StatusCode::OK);
    {
        let target_headers = builder.headers_mut().expect("headers exist");
        copy_response_headers(target_headers, &headers, false);
        target_headers.insert(CONTENT_LENGTH, HeaderValue::from(bytes.len()));
    }
    builder
        .body(Body::from(bytes))
        .expect("original bytes response is valid")
}

async fn cache_cleanup_loop(cfg: Arc<Config>) {
    let mut interval = tokio::time::interval(cfg.cache_cleanup_interval);
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);

    loop {
        interval.tick().await;
        if let Err(err) = cleanup_cache_once(&cfg).await {
            warn!(error = %err, "cache cleanup failed");
        }
    }
}

async fn cleanup_cache_once(cfg: &Config) -> anyhow::Result<()> {
    let mut files = collect_cache_files(&cfg.cache_dir).await?;
    let now = SystemTime::now();
    let mut total_bytes = files.iter().map(|file| file.len).sum::<u64>();
    let mut removed_files = 0_u64;
    let mut removed_bytes = 0_u64;

    let mut retained = Vec::with_capacity(files.len());
    for file in files.drain(..) {
        if is_older_than(file.modified, now, cfg.cache_max_age) {
            match fs::remove_file(&file.path).await {
                Ok(()) => {
                    total_bytes = total_bytes.saturating_sub(file.len);
                    removed_files += 1;
                    removed_bytes += file.len;
                }
                Err(err) => warn!(
                    path = %file.path.display(),
                    error = %err,
                    "failed to remove expired cache file"
                ),
            }
        } else {
            retained.push(file);
        }
    }

    if total_bytes > cfg.cache_max_bytes {
        retained.sort_by_key(|file| file.modified);
        for file in retained {
            if total_bytes <= cfg.cache_max_bytes {
                break;
            }
            match fs::remove_file(&file.path).await {
                Ok(()) => {
                    total_bytes = total_bytes.saturating_sub(file.len);
                    removed_files += 1;
                    removed_bytes += file.len;
                }
                Err(err) => warn!(
                    path = %file.path.display(),
                    error = %err,
                    "failed to remove cache file for size limit"
                ),
            }
        }
    }

    if removed_files > 0 {
        info!(
            removed_files,
            removed_bytes,
            remaining_bytes = total_bytes,
            "cache cleanup completed"
        );
    }

    Ok(())
}

async fn collect_cache_files(cache_dir: &Path) -> anyhow::Result<Vec<CacheFile>> {
    let mut files = Vec::new();
    for format in [TargetFormat::Avif, TargetFormat::Webp] {
        let format_dir = cache_dir.join(format.extension());
        collect_cache_files_for_format(&format_dir, format.extension(), &mut files).await?;
    }
    Ok(files)
}

async fn collect_cache_files_for_format(
    root: &Path,
    extension: &str,
    files: &mut Vec<CacheFile>,
) -> anyhow::Result<()> {
    if fs::metadata(root).await.is_err() {
        return Ok(());
    }

    let mut pending = vec![root.to_path_buf()];
    while let Some(dir) = pending.pop() {
        let mut entries = match fs::read_dir(&dir).await {
            Ok(entries) => entries,
            Err(err) => {
                warn!(path = %dir.display(), error = %err, "failed to read cache directory");
                continue;
            }
        };

        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            let metadata = match entry.metadata().await {
                Ok(metadata) => metadata,
                Err(err) => {
                    warn!(path = %path.display(), error = %err, "failed to read cache metadata");
                    continue;
                }
            };

            if metadata.is_dir() {
                pending.push(path);
                continue;
            }

            if !metadata.is_file() || !is_cache_managed_file(&path, extension) {
                continue;
            }

            files.push(CacheFile {
                path,
                len: metadata.len(),
                modified: metadata.modified().unwrap_or(SystemTime::UNIX_EPOCH),
            });
        }
    }

    Ok(())
}

fn is_cache_managed_file(path: &Path, extension: &str) -> bool {
    let Some(file_name) = path.file_name().and_then(OsStr::to_str) else {
        return false;
    };
    let expected_suffix = format!(".{extension}");
    file_name.ends_with(&expected_suffix) || file_name.contains(".tmp-")
}

fn is_older_than(modified: SystemTime, now: SystemTime, max_age: Duration) -> bool {
    now.duration_since(modified)
        .map(|age| age > max_age)
        .unwrap_or(false)
}

impl Config {
    fn from_env() -> anyhow::Result<Self> {
        let listen_addr = env::var("LISTEN_ADDR")
            .unwrap_or_else(|_| DEFAULT_LISTEN_ADDR.to_string())
            .parse::<SocketAddr>()
            .context("LISTEN_ADDR must be host:port, e.g. 127.0.0.1:9000")?;

        let upstream_base = Url::parse(
            &env::var("UPSTREAM_BASE").unwrap_or_else(|_| DEFAULT_UPSTREAM_BASE.to_string()),
        )
        .context("UPSTREAM_BASE must be an absolute URL")?;

        let cache_dir =
            PathBuf::from(env::var("CACHE_DIR").unwrap_or_else(|_| DEFAULT_CACHE_DIR.to_string()));
        let max_transform_bytes =
            parse_env_u64("MAX_TRANSFORM_BYTES", DEFAULT_MAX_TRANSFORM_BYTES)?;
        let max_pixels = parse_env_nonzero_u64("MAX_PIXELS", DEFAULT_MAX_PIXELS)?;
        let avif_quality = parse_env_f32("AVIF_QUALITY", DEFAULT_AVIF_QUALITY)?;
        let avif_speed = parse_env_u8("AVIF_SPEED", DEFAULT_AVIF_SPEED)?;
        let webp_quality = parse_env_f32("WEBP_QUALITY", DEFAULT_WEBP_QUALITY)?;
        let cache_version =
            env::var("CACHE_VERSION").unwrap_or_else(|_| DEFAULT_CACHE_VERSION.to_string());
        let enable_accept_negotiation = parse_env_bool("ENABLE_ACCEPT_NEGOTIATION", false)?;
        let default_encodes = std::thread::available_parallelism()
            .map(|n| std::cmp::max(1, n.get() / 2))
            .unwrap_or(1);
        let max_concurrent_encodes = parse_env_usize("MAX_CONCURRENT_ENCODES", default_encodes)?;
        let upstream_connect_timeout = Duration::from_secs(parse_env_nonzero_u64(
            "UPSTREAM_CONNECT_TIMEOUT_SECS",
            DEFAULT_UPSTREAM_CONNECT_TIMEOUT_SECS,
        )?);
        let upstream_request_timeout = Duration::from_secs(parse_env_nonzero_u64(
            "UPSTREAM_REQUEST_TIMEOUT_SECS",
            DEFAULT_UPSTREAM_REQUEST_TIMEOUT_SECS,
        )?);
        let cache_cleanup_enabled =
            parse_env_bool("CACHE_CLEANUP_ENABLED", DEFAULT_CACHE_CLEANUP_ENABLED)?;
        let cache_max_bytes = parse_env_nonzero_u64("CACHE_MAX_BYTES", DEFAULT_CACHE_MAX_BYTES)?;
        let cache_max_age = Duration::from_secs(
            parse_env_nonzero_u64("CACHE_MAX_AGE_DAYS", DEFAULT_CACHE_MAX_AGE_DAYS)?
                .saturating_mul(24 * 60 * 60),
        );
        let cache_cleanup_interval = Duration::from_secs(parse_env_nonzero_u64(
            "CACHE_CLEANUP_INTERVAL_SECS",
            DEFAULT_CACHE_CLEANUP_INTERVAL_SECS,
        )?);

        Ok(Self {
            listen_addr,
            upstream_base,
            cache_dir,
            max_transform_bytes,
            max_pixels,
            avif_quality,
            avif_speed,
            webp_quality,
            cache_version,
            enable_accept_negotiation,
            max_concurrent_encodes,
            upstream_connect_timeout,
            upstream_request_timeout,
            cache_cleanup_enabled,
            cache_max_bytes,
            cache_max_age,
            cache_cleanup_interval,
        })
    }
}

fn parse_env_u64(name: &str, default: u64) -> anyhow::Result<u64> {
    env::var(name)
        .map(|value| {
            value
                .parse::<u64>()
                .with_context(|| format!("{name} must be u64"))
        })
        .unwrap_or(Ok(default))
}

fn parse_env_nonzero_u64(name: &str, default: u64) -> anyhow::Result<u64> {
    let value = parse_env_u64(name, default)?;
    if value == 0 {
        return Err(anyhow!("{name} must be greater than zero"));
    }
    Ok(value)
}

fn parse_env_usize(name: &str, default: usize) -> anyhow::Result<usize> {
    let value = env::var(name)
        .map(|value| {
            value
                .parse::<usize>()
                .with_context(|| format!("{name} must be usize"))
        })
        .unwrap_or(Ok(default))?;
    if value == 0 {
        return Err(anyhow!("{name} must be greater than zero"));
    }
    Ok(value)
}

fn parse_env_u8(name: &str, default: u8) -> anyhow::Result<u8> {
    env::var(name)
        .map(|value| {
            value
                .parse::<u8>()
                .with_context(|| format!("{name} must be u8"))
        })
        .unwrap_or(Ok(default))
}

fn parse_env_f32(name: &str, default: f32) -> anyhow::Result<f32> {
    env::var(name)
        .map(|value| {
            value
                .parse::<f32>()
                .with_context(|| format!("{name} must be number"))
        })
        .unwrap_or(Ok(default))
}

fn parse_env_bool(name: &str, default: bool) -> anyhow::Result<bool> {
    match env::var(name) {
        Ok(value) => match value.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => Ok(true),
            "0" | "false" | "no" | "off" => Ok(false),
            _ => Err(anyhow!("{name} must be boolean")),
        },
        Err(_) => Ok(default),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::Request as AxumRequest;
    use http_body_util::BodyExt;
    use std::convert::Infallible;
    use tokio::net::TcpListener;
    use tower::ServiceExt;

    fn test_config(enable_accept_negotiation: bool) -> Config {
        Config {
            listen_addr: "127.0.0.1:9000".parse().unwrap(),
            upstream_base: Url::parse("http://127.0.0.1:9100").unwrap(),
            cache_dir: PathBuf::from(r"C:\minio_img_cache"),
            max_transform_bytes: DEFAULT_MAX_TRANSFORM_BYTES,
            max_pixels: DEFAULT_MAX_PIXELS,
            avif_quality: DEFAULT_AVIF_QUALITY,
            avif_speed: DEFAULT_AVIF_SPEED,
            webp_quality: DEFAULT_WEBP_QUALITY,
            cache_version: DEFAULT_CACHE_VERSION.to_string(),
            enable_accept_negotiation,
            max_concurrent_encodes: 1,
            upstream_connect_timeout: Duration::from_secs(DEFAULT_UPSTREAM_CONNECT_TIMEOUT_SECS),
            upstream_request_timeout: Duration::from_secs(DEFAULT_UPSTREAM_REQUEST_TIMEOUT_SECS),
            cache_cleanup_enabled: DEFAULT_CACHE_CLEANUP_ENABLED,
            cache_max_bytes: DEFAULT_CACHE_MAX_BYTES,
            cache_max_age: Duration::from_secs(DEFAULT_CACHE_MAX_AGE_DAYS * 24 * 60 * 60),
            cache_cleanup_interval: Duration::from_secs(DEFAULT_CACHE_CLEANUP_INTERVAL_SECS),
        }
    }

    fn test_state(cfg: Config) -> AppState {
        AppState {
            cfg: Arc::new(cfg),
            client: Client::builder()
                .redirect(reqwest::redirect::Policy::none())
                .timeout(Duration::from_secs(5))
                .build()
                .unwrap(),
            encode_limiter: Arc::new(Semaphore::new(1)),
            in_flight_encodes: Arc::new(StdMutex::new(HashMap::new())),
            temp_file_counter: Arc::new(AtomicU64::new(0)),
        }
    }

    async fn spawn_mock_upstream(
        handler: Router,
    ) -> (String, tokio::task::JoinHandle<Result<(), std::io::Error>>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let handle = tokio::spawn(async move { axum::serve(listener, handler).await });
        (format!("http://{}", addr), handle)
    }

    #[test]
    fn explicit_format_selects_transform_and_strips_format_query() {
        let uri: Uri = "/bucket/a.jpg?format=avif&v=1".parse().unwrap();
        let decision = decide_transform(&uri, &HeaderMap::new(), &test_config(false)).unwrap();

        assert_eq!(decision.target, TargetFormat::Avif);
        assert_eq!(decision.upstream_query.as_deref(), Some("v=1"));
    }

    #[test]
    fn no_format_does_not_negotiate_accept_by_default() {
        let uri: Uri = "/bucket/a.jpg".parse().unwrap();
        let mut headers = HeaderMap::new();
        headers.insert(ACCEPT, HeaderValue::from_static("image/avif,image/webp"));

        assert!(decide_transform(&uri, &headers, &test_config(false)).is_none());
    }

    #[test]
    fn accept_negotiation_can_be_enabled() {
        let uri: Uri = "/bucket/a.jpg".parse().unwrap();
        let mut headers = HeaderMap::new();
        headers.insert(
            ACCEPT,
            HeaderValue::from_static("image/avif;q=0,image/webp"),
        );

        let decision = decide_transform(&uri, &headers, &test_config(true)).unwrap();
        assert_eq!(decision.target, TargetFormat::Webp);
    }

    #[test]
    fn range_requests_bypass_transform() {
        let uri: Uri = "/bucket/a.jpg?format=webp".parse().unwrap();
        let mut headers = HeaderMap::new();
        headers.insert(RANGE, HeaderValue::from_static("bytes=0-99"));

        assert!(decide_transform(&uri, &headers, &test_config(false)).is_none());
    }

    #[test]
    fn non_jpeg_png_paths_bypass_transform() {
        let uri: Uri = "/bucket/video.mp4?format=avif".parse().unwrap();
        assert!(decide_transform(&uri, &HeaderMap::new(), &test_config(false)).is_none());
    }

    #[test]
    fn cache_key_changes_when_encoder_settings_change() {
        let uri: Uri = "/bucket/a.jpg?format=avif&v=1".parse().unwrap();
        let decision = decide_transform(&uri, &HeaderMap::new(), &test_config(false)).unwrap();
        let cfg_a = test_config(false);
        let mut cfg_b = test_config(false);
        cfg_b.avif_quality = 60.0;

        let entry_a = cache_entry(&cfg_a, &uri, &decision);
        let entry_b = cache_entry(&cfg_b, &uri, &decision);

        assert_ne!(entry_a.final_path, entry_b.final_path);
    }

    #[test]
    fn build_upstream_preserves_path_style_access() {
        let base = Url::parse("http://127.0.0.1:9100").unwrap();
        let uri: Uri = "/my-bucket/a/b.jpg?v=1".parse().unwrap();
        let url = build_upstream_url(&base, &uri).unwrap();

        assert_eq!(url.as_str(), "http://127.0.0.1:9100/my-bucket/a/b.jpg?v=1");
    }

    #[test]
    fn build_upstream_preserves_encoded_path_bytes() {
        let base = Url::parse("http://127.0.0.1:9100").unwrap();
        let uri: Uri = "/my-bucket/a%20b.jpg?v=hello%20world".parse().unwrap();
        let url = build_upstream_url(&base, &uri).unwrap();

        assert_eq!(
            url.as_str(),
            "http://127.0.0.1:9100/my-bucket/a%20b.jpg?v=hello%20world"
        );
    }

    #[test]
    fn bypass_path_strips_format_before_proxying() {
        let uri: Uri = "/my-bucket/a%20b.jpg?format=webp&v=1".parse().unwrap();
        let upstream_uri = uri_with_query(&uri, strip_format_query_raw(uri.query()).as_deref());
        let base = Url::parse("http://127.0.0.1:9100").unwrap();
        let url = build_upstream_url(&base, &upstream_uri).unwrap();

        assert_eq!(
            url.as_str(),
            "http://127.0.0.1:9100/my-bucket/a%20b.jpg?v=1"
        );
    }

    #[test]
    fn pixel_limit_rejects_oversized_images() {
        let err = ensure_pixel_limit(10_000, 10_000, 40_000_000).unwrap_err();

        assert!(err.to_string().contains("MAX_PIXELS"));
    }

    #[test]
    fn cache_temp_path_is_unique_per_write() {
        let state = test_state(test_config(false));
        let uri: Uri = "/bucket/a.jpg?format=avif".parse().unwrap();
        let decision = decide_transform(&uri, &HeaderMap::new(), &test_config(false)).unwrap();
        let entry = cache_entry(&test_config(false), &uri, &decision);

        let first = cache_temp_path(&state, &entry);
        let second = cache_temp_path(&state, &entry);

        assert_ne!(first, second);
        assert_eq!(first.parent(), entry.final_path.parent());
        assert_eq!(second.parent(), entry.final_path.parent());
    }

    #[test]
    fn cache_cleanup_only_manages_target_format_and_tmp_files() {
        assert!(is_cache_managed_file(Path::new("abc.avif"), "avif"));
        assert!(is_cache_managed_file(Path::new("abc.avif.tmp-1-2"), "avif"));
        assert!(!is_cache_managed_file(Path::new("abc.jpg"), "avif"));
        assert!(!is_cache_managed_file(Path::new("notes.txt"), "webp"));
    }

    #[tokio::test]
    async fn get_without_format_proxies_original() {
        async fn handler(req: Request) -> Result<Response<Body>, Infallible> {
            assert_eq!(req.uri().path(), "/bucket/a.jpg");
            assert_eq!(req.uri().query(), Some("v=1"));
            Ok(Response::builder()
                .status(StatusCode::OK)
                .header(CONTENT_TYPE, "image/jpeg")
                .body(Body::from("original"))
                .unwrap())
        }

        let (upstream, _handle) = spawn_mock_upstream(Router::new().fallback(any(handler))).await;
        let mut cfg = test_config(false);
        cfg.upstream_base = Url::parse(&upstream).unwrap();
        let app = build_router(test_state(cfg));

        let response = app
            .oneshot(
                AxumRequest::builder()
                    .uri("/bucket/a.jpg?v=1")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        assert_eq!(&body[..], b"original");
    }

    #[tokio::test]
    async fn transform_failure_returns_original_bytes() {
        async fn handler(req: Request) -> Result<Response<Body>, Infallible> {
            assert_eq!(req.uri().path(), "/bucket/a.jpg");
            assert_eq!(req.uri().query(), Some("v=1"));
            Ok(Response::builder()
                .status(StatusCode::OK)
                .header(CONTENT_TYPE, "image/jpeg")
                .body(Body::from("not an actual image"))
                .unwrap())
        }

        let cache_dir = tempfile::tempdir().unwrap();
        let (upstream, _handle) = spawn_mock_upstream(Router::new().fallback(any(handler))).await;
        let mut cfg = test_config(false);
        cfg.cache_dir = cache_dir.path().to_path_buf();
        cfg.upstream_base = Url::parse(&upstream).unwrap();
        let app = build_router(test_state(cfg));

        let response = app
            .oneshot(
                AxumRequest::builder()
                    .uri("/bucket/a.jpg?format=webp&v=1")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response.headers().get(CONTENT_TYPE).unwrap(),
            HeaderValue::from_static("image/jpeg")
        );
        let body = response.into_body().collect().await.unwrap().to_bytes();
        assert_eq!(&body[..], b"not an actual image");
    }

    #[tokio::test]
    async fn unsupported_method_is_rejected_before_upstream() {
        let (upstream, _handle) =
            spawn_mock_upstream(Router::new().fallback(any(|| async { "unexpected" }))).await;
        let mut cfg = test_config(false);
        cfg.upstream_base = Url::parse(&upstream).unwrap();
        let app = build_router(test_state(cfg));

        let response = app
            .oneshot(
                AxumRequest::builder()
                    .method(Method::POST)
                    .uri("/bucket/a.jpg")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::METHOD_NOT_ALLOWED);
    }

    #[tokio::test]
    async fn upstream_connection_failure_returns_bad_gateway() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        drop(listener);

        let mut cfg = test_config(false);
        cfg.upstream_base = Url::parse(&format!("http://{}", addr)).unwrap();
        cfg.upstream_connect_timeout = Duration::from_millis(50);
        cfg.upstream_request_timeout = Duration::from_millis(50);
        let app = build_router(test_state(cfg));

        let response = app
            .oneshot(
                AxumRequest::builder()
                    .uri("/bucket/a.jpg")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    }
}
