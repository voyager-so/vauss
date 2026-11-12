use super::*;

fn fixture(bytes: &[u8]) -> Asset {
    Asset {
        id: "fixture".into(),
        target: "any".into(),
        version: "1".into(),
        url: "https://example.invalid/fixture".into(),
        github_api_url: None,
        sha256: format!("{:x}", Sha256::digest(bytes)),
        bytes: bytes.len() as u64,
        format: Format::Raw,
        inner_zip: None,
        entrypoint: "model.onnx".into(),
        executable: false,
        license: "MIT".into(),
        license_url: "https://example.invalid/license".into(),
        source_url: "https://example.invalid/source".into(),
        requires_gpl_consent: false,
        note: String::new(),
    }
}

#[test]
fn pinned_catalog_is_valid_and_covers_ffmpeg_targets() {
    let assets = catalog().unwrap();
    let mut keys = std::collections::BTreeSet::new();
    for asset in &assets {
        assert!(keys.insert((&asset.id, &asset.target)));
    }
    for target in [
        "x86_64-unknown-linux-gnu",
        "aarch64-apple-darwin",
        "x86_64-pc-windows-msvc",
    ] {
        assert_eq!(
            asset_for("ffmpeg", target).unwrap().license,
            "LGPL-2.1-or-later"
        );
    }
}

#[test]
fn corrupt_and_truncated_downloads_are_rejected() {
    let asset = fixture(b"good");
    copy_verified(&b"good"[..], &mut Vec::new(), &asset).unwrap();
    assert!(
        copy_verified(&b"bad!"[..], &mut Vec::new(), &asset)
            .unwrap_err()
            .downcast_ref::<FetchError>()
            .is_some()
    );
    assert!(copy_verified(&b"goo"[..], &mut Vec::new(), &asset).is_err());
    assert!(copy_verified(&b"good-more"[..], &mut Vec::new(), &asset).is_err());
}

#[test]
fn gpl_requires_selection_even_when_cached() {
    let dir = tempfile::tempdir().unwrap();
    let mut asset = fixture(b"data");
    asset.requires_gpl_consent = true;
    let error = Fetcher::new(dir.path())
        .ensure_asset(&asset, LicenseConsent::Default)
        .unwrap_err();
    assert!(matches!(
        error.downcast_ref(),
        Some(FetchError::LicenseRequired { .. })
    ));
    assert_eq!(fs::read_dir(dir.path()).unwrap().count(), 0);
}

#[test]
fn partial_or_corrupt_cache_is_not_accepted() {
    let dir = tempfile::tempdir().unwrap();
    let asset = fixture(b"data");
    let fetcher = Fetcher::new(dir.path());
    assert!(fetcher.cached(&asset).unwrap().is_none());
    let install = dir.path().join(asset.key());
    fs::create_dir(&install).unwrap();
    fs::write(install.join(&asset.entrypoint), b"data").unwrap();
    assert!(fetcher.cached(&asset).unwrap().is_none());
    let receipt = Receipt {
        links: BTreeMap::new(),
        asset: asset.clone(),
        files: BTreeMap::from([(asset.entrypoint.clone(), asset.sha256.clone())]),
    };
    fs::write(install.join(RECEIPT), serde_json::to_vec(&receipt).unwrap()).unwrap();
    assert!(fetcher.cached(&asset).unwrap().is_some());
    fs::write(install.join(&asset.entrypoint), b"evil").unwrap();
    assert!(fetcher.cached(&asset).unwrap().is_none());
}

#[test]
fn zip_traversal_and_links_are_rejected() {
    for (name, link) in [
        ("../escape", false),
        ("/absolute", false),
        ("C:\\escape", false),
        ("dir/../escape", false),
        ("dir\\escape", false),
        ("//server/share/escape", false),
        ("link", true),
    ] {
        let dir = tempfile::tempdir().unwrap();
        let archive = dir.path().join("asset.zip");
        let mut writer = zip::ZipWriter::new(File::create(&archive).unwrap());
        let options = zip::write::SimpleFileOptions::default();
        if link {
            writer.add_symlink(name, "../escape", options).unwrap();
        } else {
            writer.start_file(name, options).unwrap();
            writer.write_all(b"evil").unwrap();
        }
        writer.finish().unwrap();
        let mut asset = fixture(b"data");
        asset.format = Format::Zip;
        let destination = dir.path().join("out");
        fs::create_dir(&destination).unwrap();
        assert!(
            unpack(&archive, &destination, &asset).is_err(),
            "accepted {name}"
        );
    }
}

#[test]
fn zip_safe_nested_paths_are_extracted() {
    let dir = tempfile::tempdir().unwrap();
    let archive = dir.path().join("asset.zip");
    let mut writer = zip::ZipWriter::new(File::create(&archive).unwrap());
    writer
        .start_file("tools/bin/helper", zip::write::SimpleFileOptions::default())
        .unwrap();
    writer.write_all(b"verified helper").unwrap();
    writer.finish().unwrap();
    let mut asset = fixture(b"data");
    asset.format = Format::Zip;
    let destination = dir.path().join("out");
    fs::create_dir(&destination).unwrap();
    unpack(&archive, &destination, &asset).unwrap();
    assert_eq!(fs::read(destination.join("tools/bin/helper")).unwrap(), b"verified helper");
}

#[test]
fn zip_disagreeing_local_and_central_names_are_rejected_before_writing() {
    let dir = tempfile::tempdir().unwrap();
    let archive = dir.path().join("asset.zip");
    let mut writer = zip::ZipWriter::new(File::create(&archive).unwrap());
    writer
        .start_file("safe.txt", zip::write::SimpleFileOptions::default())
        .unwrap();
    writer.write_all(b"untrusted payload").unwrap();
    writer.finish().unwrap();

    // The central directory still advertises safe.txt, while the local header
    // names a different file. Both names are harmless: the disagreement itself
    // must fail instead of letting two ZIP readers see different archive contents.
    let mut bytes = fs::read(&archive).unwrap();
    assert_eq!(&bytes[..4], b"PK\x03\x04");
    assert_eq!(u16::from_le_bytes([bytes[26], bytes[27]]), 8);
    bytes[30..38].copy_from_slice(b"evil.txt");
    fs::write(&archive, bytes).unwrap();
    let mut asset = fixture(b"data");
    asset.format = Format::Zip;
    let destination = dir.path().join("out");
    fs::create_dir(&destination).unwrap();
    assert!(unpack(&archive, &destination, &asset).is_err());
    assert_eq!(fs::read_dir(destination).unwrap().count(), 0);
}

#[test]
fn tar_links_are_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let archive = dir.path().join("asset.tar.gz");
    let encoder =
        flate2::write::GzEncoder::new(File::create(&archive).unwrap(), flate2::Compression::fast());
    let mut tar = tar::Builder::new(encoder);
    let mut header = tar::Header::new_gnu();
    header.set_size(0);
    header.set_mode(0o777);
    header.set_entry_type(tar::EntryType::Symlink);
    tar.append_link(&mut header, "link", "../outside").unwrap();
    tar.into_inner().unwrap().finish().unwrap();
    let mut asset = fixture(b"data");
    asset.format = Format::TarGz;
    let destination = dir.path().join("out");
    fs::create_dir(&destination).unwrap();
    assert!(unpack(&archive, &destination, &asset).is_err());
}

#[cfg(unix)]
#[test]
fn framework_links_are_preserved_and_checked() {
    let dir = tempfile::tempdir().unwrap();
    let asset = fixture(b"data");
    let archive = dir.path().join("framework.zip");
    let mut writer = zip::ZipWriter::new(File::create(&archive).unwrap());
    let options = zip::write::SimpleFileOptions::default();
    writer
        .add_symlink("Framework/lib", "Versions/Current/lib", options)
        .unwrap();
    writer
        .add_symlink("Framework/Versions/Current", "A", options)
        .unwrap();
    writer
        .start_file("Framework/Versions/A/lib", options)
        .unwrap();
    writer.write_all(b"data").unwrap();
    writer.finish().unwrap();
    let mut asset = asset;
    asset.format = Format::Zip;
    asset.entrypoint = "Framework/Versions/A/lib".into();
    let install = dir.path().join(asset.key());
    fs::create_dir(&install).unwrap();
    unpack(&archive, &install, &asset).unwrap();
    assert_eq!(fs::read(install.join("Framework/lib")).unwrap(), b"data");
    let mut files = BTreeMap::new();
    let mut links = BTreeMap::new();
    inventory(&install, &install, &mut files, &mut links).unwrap();
    assert_eq!(links.len(), 2);
    let receipt = Receipt {
        asset: asset.clone(),
        files,
        links,
    };
    fs::write(install.join(RECEIPT), serde_json::to_vec(&receipt).unwrap()).unwrap();
    let fetcher = Fetcher::new(dir.path());
    assert!(fetcher.cached(&asset).unwrap().is_some());
    fs::remove_file(install.join("Framework/lib")).unwrap();
    std::os::unix::fs::symlink("/tmp", install.join("Framework/lib")).unwrap();
    assert!(fetcher.cached(&asset).unwrap().is_none());
}

#[test]
fn configured_ffmpeg_precedes_voyager_and_does_not_fetch() {
    let dir = tempfile::tempdir().unwrap();
    let configured = dir.path().join("custom");
    fs::write(&configured, b"test").unwrap();
    let fetcher = Fetcher::new(dir.path().join("cache"));
    assert_eq!(
        fetcher
            .ffmpeg(Some(&configured), Some(Path::new("missing")))
            .unwrap(),
        configured
    );
    assert!(fetcher.ffmpeg(Some(Path::new("missing")), None).is_err());
    assert!(!dir.path().join("cache").exists());
}

#[test]
fn archive_size_limit_is_enforced_before_writing() {
    let dir = tempfile::tempdir().unwrap();
    let mut remaining = 3;
    let file = dir.path().join("entry");
    assert!(write_entry(&mut &b"data"[..], &file, 4, &mut remaining).is_err());
    assert!(!file.exists());
}

#[test]
fn private_asset_errors_name_remedies_and_untrusted_auth_hosts_are_rejected() {
    let mut asset = fixture(b"data");
    asset.github_api_url =
        Some("https://api.github.com/repos/voyager-so/vauss/releases/assets/123".into());
    asset.validate().unwrap();
    // GitHub answers an anonymous download of a non-public release asset with 404.
    assert!(needs_token(reqwest::StatusCode::NOT_FOUND));
    assert!(needs_token(reqwest::StatusCode::UNAUTHORIZED));
    assert!(!needs_token(reqwest::StatusCode::INTERNAL_SERVER_ERROR));
    let message = FetchError::AuthenticationRequired { id: "ffmpeg-public".into() }.to_string();
    for remedy in ["SPLAT_GITHUB_TOKEN", "tools/ffmpeg-public/", "SPLAT_FFMPEG"] {
        assert!(message.contains(remedy), "{message}");
    }
    for url in [
        "https://example.invalid/repos/voyager-so/vauss/releases/assets/123",
        "https://api.github.com/user",
        "https://api.github.com/repos/voyager-so/vauss/releases/assets/123?token=secret",
        "https://user@api.github.com/repos/voyager-so/vauss/releases/assets/123",
    ] {
        asset.github_api_url = Some(url.into());
        assert!(asset.validate().is_err());
    }
}

/// Serves `routes` (path -> status and body) on a local port, recording each request's path and
/// Authorization header, for `requests` requests.
fn serve(
    routes: &'static [(&'static str, u16, &'static [u8])],
    requests: usize,
) -> (String, std::sync::mpsc::Receiver<(String, Option<String>)>) {
    use std::io::{BufRead, BufReader, Write};
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let (seen, log) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        for stream in listener.incoming().take(requests) {
            let mut stream = stream.unwrap();
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let mut line = String::new();
            reader.read_line(&mut line).unwrap();
            let path = line.split(' ').nth(1).unwrap_or_default().to_string();
            let mut auth = None;
            loop {
                let mut header = String::new();
                reader.read_line(&mut header).unwrap();
                if header.trim().is_empty() {
                    break;
                }
                if let Some((name, value)) = header.split_once(':')
                    && name.eq_ignore_ascii_case("authorization")
                {
                    auth = Some(value.trim().to_string());
                }
            }
            let (status, body) = routes
                .iter()
                .find(|(p, _, _)| *p == path)
                .map_or((404, &b""[..]), |(_, s, b)| (*s, *b));
            write!(stream, "HTTP/1.1 {status} X\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", body.len()).unwrap();
            stream.write_all(body).unwrap();
            seen.send((path, auth)).unwrap();
        }
    });
    (base, log)
}

fn served_asset(base: &str) -> Asset {
    let mut asset = fixture(b"data");
    asset.url = format!("{base}/public");
    asset.github_api_url = Some(format!("{base}/api"));
    asset
}

fn body(response: reqwest::blocking::Response) -> Vec<u8> {
    response.bytes().unwrap().to_vec()
}

#[test]
fn public_download_needs_no_token_even_when_one_is_set() {
    let (base, log) = serve(&[("/public", 200, b"data")], 1);
    let client = reqwest::blocking::Client::new();
    let response = open_download(&client, &served_asset(&base), Some("expired")).unwrap();
    assert_eq!(body(response), b"data");
    assert_eq!(log.recv().unwrap(), ("/public".to_string(), None));
}

#[test]
fn non_public_release_falls_back_to_the_api_with_the_token() {
    let (base, log) = serve(&[("/public", 404, b""), ("/api", 200, b"data")], 2);
    let client = reqwest::blocking::Client::new();
    let response = open_download(&client, &served_asset(&base), Some("secret")).unwrap();
    assert_eq!(body(response), b"data");
    assert_eq!(log.recv().unwrap(), ("/public".to_string(), None));
    assert_eq!(log.recv().unwrap(), ("/api".to_string(), Some("Bearer secret".to_string())));
}

#[test]
fn non_public_release_without_a_usable_token_says_what_to_do() {
    let client = reqwest::blocking::Client::new();
    let (base, log) = serve(&[("/public", 404, b"")], 1);
    let err = open_download(&client, &served_asset(&base), None).unwrap_err();
    assert!(matches!(err.downcast_ref(), Some(FetchError::AuthenticationRequired { .. })));
    assert!(err.to_string().contains("SPLAT_GITHUB_TOKEN"), "{err}");
    assert_eq!(log.recv().unwrap().0, "/public");
    let (base, _log) = serve(&[("/public", 404, b""), ("/api", 401, b"")], 2);
    let err = open_download(&client, &served_asset(&base), Some("expired")).unwrap_err();
    assert!(matches!(err.downcast_ref(), Some(FetchError::AuthenticationRequired { .. })));
    assert!(err.to_string().contains("SPLAT_GITHUB_TOKEN was refused"), "{err}");
    // A failure that is not about access stays that failure.
    let (base, _log) = serve(&[("/public", 500, b"")], 1);
    let err = open_download(&client, &served_asset(&base), Some("secret")).unwrap_err();
    assert!(err.downcast_ref::<FetchError>().is_none(), "{err:#}");
}
