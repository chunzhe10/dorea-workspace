#!/bin/bash
# Shared functions for devcontainer scripts.
# Source this file; do not execute directly.

err() { echo "Error: $*" >&2; }

# Retry a command up to N times with exponential backoff.
# Usage: retry <max_attempts> command args...
retry() {
    local max="$1"; shift
    local attempt=1 delay=5
    while true; do
        if "$@"; then
            return 0
        fi
        if [ "$attempt" -ge "$max" ]; then
            err "Failed after $max attempts: $*"
            return 1
        fi
        echo "  Attempt $attempt/$max failed, retrying in ${delay}s..."
        sleep "$delay"
        attempt=$((attempt + 1))
        delay=$((delay * 2))
    done
}

# Print status while a background command runs.
# Usage: spin "message" command args...
# On failure, shows captured output for debugging.
spin() {
    local msg="$1"; shift
    local logfile
    logfile=$(mktemp)
    "$@" >"$logfile" 2>&1 &
    local pid=$!
    printf "  %s " "$msg"
    while kill -0 "$pid" 2>/dev/null; do
        printf "."
        sleep 1
    done
    if wait "$pid"; then
        echo " done"
    else
        echo " FAILED"
        if [ -s "$logfile" ]; then
            echo "--- output from: $* ---" >&2
            cat "$logfile" >&2
            echo "--- end output ---" >&2
        fi
        rm -f "$logfile"
        return 1
    fi
    rm -f "$logfile"
}

# Detect if running inside WSL.
is_wsl() {
    grep -qi "microsoft\|wsl" /proc/version 2>/dev/null
}

# Detect if GPU (NVIDIA) is available inside the container.
has_gpu() {
    command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1
}

WORKSPACE_ROOT="${CORVIA_WORKSPACE:-$(pwd)}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Ensure uv/uvx are on PATH even if the Dockerfile mv to /usr/local/bin failed.
if [ -d "/root/.local/bin" ] && ! echo "$PATH" | grep -q "/root/.local/bin"; then
    export PATH="/root/.local/bin:$PATH"
fi

# Detect architecture suffix for binary downloads.
detect_arch() {
    local arch
    arch="$(uname -m)"
    case "$arch" in
        x86_64)  echo "amd64" ;;
        aarch64) echo "arm64" ;;
        *)       err "Unsupported architecture: $arch"; return 1 ;;
    esac
}

# Wait for network connectivity (up to 60s).
wait_for_network() {
    printf "  Waiting for network "
    for _attempt in $(seq 1 30); do
        if curl -fsL --max-time 2 -o /dev/null https://github.com 2>/dev/null; then
            echo " ready"
            return 0
        fi
        printf "."
        sleep 2
    done
    if ! curl -fsL --max-time 2 -o /dev/null https://github.com 2>/dev/null; then
        err "Network not available after 60 seconds."
        return 1
    fi
}

# Download and install corvia + corvia-inference binaries.
# Caches the installed release tag in /usr/local/share/corvia-release-tag.
# Skips download if the cached tag matches the latest release.
install_binaries() {
    local arch_suffix
    arch_suffix="$(detect_arch)" || return 1
    local gh_repo="chunzhe10/corvia"
    local install_dir="/usr/local/bin"
    local tag_file="/usr/local/share/corvia-release-tag"
    local cli_asset="corvia-cli-linux-${arch_suffix}"
    local inf_asset="corvia-inference-linux-${arch_suffix}"
    local adp_basic_asset="corvia-adapter-basic-linux-${arch_suffix}"
    local adp_git_asset="corvia-adapter-git-linux-${arch_suffix}"

    # Check latest release tag
    local latest_tag=""
    if command -v gh >/dev/null 2>&1 && gh auth status >/dev/null 2>&1; then
        latest_tag=$(gh release view --repo "$gh_repo" --json tagName -q .tagName 2>/dev/null || true)
    else
        latest_tag=$(curl -fsL --max-time 5 "https://api.github.com/repos/$gh_repo/releases/latest" 2>/dev/null \
            | python3 -c "import sys,json; print(json.load(sys.stdin).get('tag_name',''))" 2>/dev/null || true)
    fi

    # Skip if already on latest
    if [ -n "$latest_tag" ] && [ -f "$tag_file" ] && [ "$(cat "$tag_file")" = "$latest_tag" ] \
        && [ -x "$install_dir/corvia" ] && [ -x "$install_dir/corvia-inference" ] \
        && [ -x "$install_dir/corvia-adapter-basic" ] && [ -x "$install_dir/corvia-adapter-git" ]; then
        echo "    binaries up to date ($latest_tag)"
        return 0
    fi

    if [ -n "$latest_tag" ]; then
        echo "    downloading $latest_tag"
    else
        echo "    downloading latest release"
    fi

    local tmpdir
    tmpdir=$(mktemp -d)

    if command -v gh >/dev/null 2>&1 && gh auth status >/dev/null 2>&1; then
        spin "    fetching (gh)..." \
            gh release download --repo "$gh_repo" --pattern "$cli_asset" --pattern "$inf_asset" \
                --pattern "$adp_basic_asset" --pattern "$adp_git_asset" \
                --pattern "libonnxruntime_providers_*-linux-${arch_suffix}.so" --dir "$tmpdir"
    else
        local url="https://github.com/$gh_repo/releases/latest/download"
        # Required binary downloads (must succeed)
        local pids=()
        for asset in "$cli_asset" "$inf_asset" "$adp_basic_asset" "$adp_git_asset"; do
            curl -fsL --retry 3 --retry-delay 2 -o "$tmpdir/$asset" "$url/$asset" 2>/dev/null &
            pids+=($!)
        done
        # Best-effort ORT provider .so downloads (may not exist in older releases)
        for asset in "libonnxruntime_providers_shared-linux-${arch_suffix}.so" \
            "libonnxruntime_providers_cuda-linux-${arch_suffix}.so" \
            "libonnxruntime_providers_openvino-linux-${arch_suffix}.so"; do
            curl -fsL --retry 1 -o "$tmpdir/$asset" "$url/$asset" 2>/dev/null &
            pids+=($!)
        done
        printf "    fetching (curl)..."
        while :; do
            alive=false
            for pid in "${pids[@]}"; do
                if kill -0 "$pid" 2>/dev/null; then
                    alive=true
                    break
                fi
            done
            [ "$alive" = true ] || break
            printf "."
            sleep 1
        done
        # Only check required binaries (first 4 pids)
        local ok=true
        local i=0
        for pid in "${pids[@]}"; do
            if [ "$i" -lt 4 ]; then
                wait "$pid" || ok=false
            else
                wait "$pid" 2>/dev/null || true  # ORT libs are best-effort
            fi
            i=$((i + 1))
        done
        if [ "$ok" = true ]; then
            echo " done"
        else
            echo " FAILED"
            rm -rf "$tmpdir"
            return 1
        fi
    fi

    sudo cp "$tmpdir/$cli_asset" "$install_dir/corvia"
    sudo cp "$tmpdir/$inf_asset" "$install_dir/corvia-inference"
    sudo cp "$tmpdir/$adp_basic_asset" "$install_dir/corvia-adapter-basic"
    sudo cp "$tmpdir/$adp_git_asset" "$install_dir/corvia-adapter-git"
    sudo chmod +x "$install_dir/corvia" "$install_dir/corvia-inference" \
        "$install_dir/corvia-adapter-basic" "$install_dir/corvia-adapter-git"

    # Install ORT provider shared libraries (best-effort: skip if not in release)
    local ort_lib_dir="/usr/lib/x86_64-linux-gnu"
    local ort_libs_installed=false
    for lib in libonnxruntime_providers_shared libonnxruntime_providers_cuda libonnxruntime_providers_openvino; do
        local lib_asset="${lib}-linux-${arch_suffix}.so"
        local lib_file="$tmpdir/$lib_asset"
        if [ -f "$lib_file" ]; then
            sudo cp "$lib_file" "$ort_lib_dir/${lib}.so"
            ort_libs_installed=true
        fi
    done
    if [ "$ort_libs_installed" = true ]; then
        sudo ldconfig
    fi

    rm -rf "$tmpdir"

    # Cache the installed tag
    if [ -n "$latest_tag" ]; then
        echo "$latest_tag" | sudo tee "$tag_file" >/dev/null
    fi
}

# Download the latest VS Code extension VSIX from workspace releases.
# Fallback for when vsce is unavailable to build from source.
install_extension() {
    local gh_repo="chunzhe10/corvia-workspace"
    local ext_dir="$WORKSPACE_ROOT/.devcontainer/extensions/corvia-services"
    local tmpdir
    tmpdir=$(mktemp -d)

    if command -v gh >/dev/null 2>&1 && gh auth status >/dev/null 2>&1; then
        gh release download --repo "$gh_repo" --pattern "corvia-services-*.vsix" --dir "$tmpdir" 2>/dev/null || { rm -rf "$tmpdir"; return 1; }
    else
        # List release assets via GitHub API, download the first VSIX match
        local url="https://api.github.com/repos/$gh_repo/releases/latest"
        local asset_url
        asset_url=$(curl -fsL "$url" 2>/dev/null \
            | python3 -c "import sys,json; assets=json.load(sys.stdin).get('assets',[]); vsix=[a for a in assets if a['name'].endswith('.vsix')]; print(vsix[0]['browser_download_url'] if vsix else '')" 2>/dev/null)
        if [ -z "$asset_url" ]; then
            rm -rf "$tmpdir"
            return 1
        fi
        curl -fsL --retry 3 --retry-delay 2 -o "$tmpdir/corvia-services.vsix" "$asset_url" 2>/dev/null \
            || { rm -rf "$tmpdir"; return 1; }
    fi

    local vsix
    vsix=$(ls -t "$tmpdir"/*.vsix 2>/dev/null | head -1)
    if [ -n "$vsix" ] && [ -f "$vsix" ]; then
        mkdir -p "$ext_dir"
        cp "$vsix" "$ext_dir/"
        echo "  Downloaded extension: $(basename "$vsix")"
    fi
    rm -rf "$tmpdir"
}

# Install a local VS Code extension by extracting a .vsix into the extensions dir.
# This bypasses the code CLI entirely — works before VS Code's IPC socket is ready.
# Usage: install_vsix_direct <vsix_path>
install_vsix_direct() {
    local vsix_path="$1"
    if [ ! -f "$vsix_path" ]; then
        err "VSIX not found: $vsix_path"
        return 1
    fi

    # Read publisher and name/version from package.json inside the .vsix (it's a zip)
    local pkg_json
    pkg_json=$(VSIX_PATH="$vsix_path" python3 -c "
import zipfile, json, sys, os
with zipfile.ZipFile(os.environ['VSIX_PATH']) as z:
    with z.open('extension/package.json') as f:
        d = json.load(f)
        print(json.dumps({'publisher': d['publisher'], 'name': d['name'], 'version': d['version']}))
" 2>/dev/null) || { err "Failed to read package.json from VSIX"; return 1; }

    local publisher name version
    publisher=$(echo "$pkg_json" | python3 -c "import sys,json; print(json.load(sys.stdin)['publisher'])")
    name=$(echo "$pkg_json" | python3 -c "import sys,json; print(json.load(sys.stdin)['name'])")
    version=$(echo "$pkg_json" | python3 -c "import sys,json; print(json.load(sys.stdin)['version'])")

    local ext_id="${publisher}.${name}-${version}"

    # Detect VS Code server variant: insiders vs stable
    local server_dirs=()
    if [ -d "/root/.vscode-server-insiders" ]; then
        server_dirs+=("/root/.vscode-server-insiders/extensions")
    fi
    if [ -d "/root/.vscode-server" ]; then
        server_dirs+=("/root/.vscode-server/extensions")
    fi
    # Fallback if neither exists yet (pre-first-connection)
    if [ ${#server_dirs[@]} -eq 0 ]; then
        server_dirs=("/root/.vscode-server/extensions")
    fi

    local installed=0
    for ext_parent in "${server_dirs[@]}"; do
        local ext_dir="${ext_parent}/${ext_id}"

        if [ -d "$ext_dir" ] && [ -f "$ext_dir/package.json" ]; then
            echo "    $ext_id already installed in $ext_parent"
            installed=1
            continue
        fi

        # Extract extension/ contents from the .vsix into the target directory
        local tmpdir
        tmpdir=$(mktemp -d)
        VSIX_PATH="$vsix_path" EXTRACT_DIR="$tmpdir" python3 -c "
import zipfile, os
with zipfile.ZipFile(os.environ['VSIX_PATH']) as z:
    for info in z.infolist():
        if info.filename.startswith('extension/'):
            # Strip 'extension/' prefix
            rel = info.filename[len('extension/'):]
            if not rel:
                continue
            target = os.path.join(os.environ['EXTRACT_DIR'], rel)
            if info.is_dir():
                os.makedirs(target, exist_ok=True)
            else:
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with z.open(info) as src, open(target, 'wb') as dst:
                    dst.write(src.read())
" 2>/dev/null || { rm -rf "$tmpdir"; err "Failed to extract VSIX"; return 1; }

        mkdir -p "$ext_parent"
        rm -rf "$ext_dir"
        mv "$tmpdir" "$ext_dir"
        echo "    installed $ext_id in $ext_parent"
        installed=1
    done

    [ "$installed" -eq 1 ] || { err "No VS Code server directory found"; return 1; }
}

# Ensure corvia binary is available, installing if needed.
ensure_corvia() {
    if [ -x "/usr/local/bin/corvia" ]; then
        return 0
    fi
    echo "corvia binary not found — installing..."
    wait_for_network || return 1
    echo "Installing Corvia binaries..."
    retry 3 install_binaries
}

# Fix ownership of workspace files so the current user can write.
# Needed when workspace is mounted/cloned as root but scripts run as vscode.
# Uses find to only chown files not already owned by the current user,
# avoiding a slow recursive chown on every start.
fix_workspace_perms() {
    local uid
    uid="$(id -u)"
    if find "$WORKSPACE_ROOT" -maxdepth 1 -not -user "$uid" -print -quit 2>/dev/null | grep -q .; then
        echo "  Fixing workspace permissions..."
        sudo chown -R "$uid:$(id -g)" "$WORKSPACE_ROOT"
    fi
}

# Clone a repo into a directory that may already contain subdirectories
# (e.g. from Docker volume mounts). Falls back to git init + pull when
# the target directory is non-empty.
# Usage: clone_into_nonempty <url> <dest>
clone_into_nonempty() {
    local url="$1" dest="$2"
    if [ -d "$dest/.git" ]; then
        return 0  # already cloned
    fi
    if [ ! -d "$dest" ] || [ -z "$(ls -A "$dest" 2>/dev/null)" ]; then
        git clone "$url" "$dest"
    else
        # Directory exists and is non-empty (e.g. Docker volume mount created it).
        # Clone to temp, then move .git in and checkout.
        local tmpdir
        tmpdir=$(mktemp -d)
        git clone "$url" "$tmpdir"
        mv "$tmpdir/.git" "$dest/.git"
        rm -rf "$tmpdir"
        git -C "$dest" checkout -- .
    fi
}

# Pre-clone repos listed in corvia.toml before `corvia workspace init`,
# which fails if the target directory is non-empty.
pre_clone_repos() {
    local repos_dir="$WORKSPACE_ROOT/repos"
    # Parse repos from corvia.toml
    python3 -c "
import tomllib, json, sys
with open('$WORKSPACE_ROOT/corvia.toml', 'rb') as f:
    cfg = tomllib.load(f)
for r in cfg.get('workspace', {}).get('repos', []):
    print(json.dumps({'name': r['name'], 'url': r['url']}))
" 2>/dev/null | while IFS= read -r line; do
        local name url dest
        name=$(echo "$line" | python3 -c "import sys,json; print(json.load(sys.stdin)['name'])")
        url=$(echo "$line" | python3 -c "import sys,json; print(json.load(sys.stdin)['url'])")
        dest="$repos_dir/$name"
        if [ ! -d "$dest/.git" ]; then
            echo "  Cloning $name..."
            clone_into_nonempty "$url" "$dest" || err "Failed to clone $name"
        fi
    done
}

# Initialize the corvia workspace.
# Idempotent: skips `corvia workspace init` if the store already exists.
init_workspace() {
    fix_workspace_perms
    pre_clone_repos
    if [ ! -f "$WORKSPACE_ROOT/.corvia/lite_store.redb" ]; then
        spin "Initializing workspace..." corvia workspace init
    else
        echo "    workspace already initialized"
    fi
}

# Install a Python package in editable mode using uv.
# Uses sudo because /usr/local/lib/python3.x/ is root-owned.
install_python_editable() {
    local pkg_path="$1"
    if ! command -v uv >/dev/null 2>&1; then
        err "uv not found. Install uv: curl -LsSf https://astral.sh/uv/install.sh | sh"
        return 1
    fi
    sudo uv pip install --system --break-system-packages -e "$pkg_path" --quiet
}


# Forward GitHub CLI credentials from host mount.
# Copies host config into the container's ~/.config/gh if the host has valid auth
# and the container doesn't, or if the host config is newer.
forward_gh_auth() {
    local host_dir="/root/.config/gh-host"
    local local_dir="/root/.config/gh"
    local host_hosts="$host_dir/hosts.yml"
    local local_hosts="$local_dir/hosts.yml"

    # No host mount or empty — nothing to forward
    if [ ! -d "$host_dir" ] || [ ! -f "$host_hosts" ]; then
        echo "  gh: no host credentials found (run 'gh auth login' on your host machine)"
        return 0
    fi

    # Validate host config has actual content (not just an empty file)
    if [ ! -s "$host_hosts" ]; then
        echo "  gh: host credentials file is empty — skipping"
        return 0
    fi

    # Copy if container has no auth, or host is newer
    if [ ! -f "$local_hosts" ] || [ "$host_hosts" -nt "$local_hosts" ]; then
        mkdir -p "$local_dir"
        cp "$host_hosts" "$local_hosts"
        # Also copy config.yml if present (protocol preferences, etc.)
        [ -f "$host_dir/config.yml" ] && cp "$host_dir/config.yml" "$local_dir/config.yml"
        echo "  gh: forwarded credentials from host"
    else
        echo "  gh: credentials already up to date"
    fi

    # Verify auth works (retry up to 3 times — token refresh can race)
    local auth_ok=false
    for _gh_attempt in 1 2 3; do
        if gh auth status >/dev/null 2>&1; then
            auth_ok=true
            break
        fi
        if [ "$_gh_attempt" -lt 3 ]; then
            echo "  gh: auth check failed, retrying (${_gh_attempt}/3)..."
            sleep 2
            # Re-copy in case host refreshed the token
            cp "$host_hosts" "$local_hosts"
        fi
    done
    if [ "$auth_ok" = true ]; then
        local gh_user
        gh_user=$(gh api user --jq .login 2>/dev/null || echo "unknown")
        echo "  gh: authenticated as $gh_user"
    else
        err "gh: forwarded credentials are invalid — run 'gh auth login' to re-authenticate"
    fi
}

# Ensure Claude Code credentials are available.
# Primary: direct bind mount of ~/.claude (read-write, tokens can refresh).
# Fallback: copy from read-only .claude-host mount if direct mount failed.
# Last resort: prompt user to authenticate manually.
forward_claude_auth() {
    local creds="/root/.claude/.credentials.json"

    # Check if direct bind mount is working (has valid credentials)
    if [ -f "$creds" ] && python3 -c "import json; json.load(open('$creds'))" 2>/dev/null; then
        echo "  claude: credentials available (bind mount)"
        return 0
    fi

    # Fallback: copy from read-only host mount if available
    local host_creds="/root/.claude-host/.credentials.json"
    if [ -f "$host_creds" ] && python3 -c "import json; json.load(open('$host_creds'))" 2>/dev/null; then
        mkdir -p /root/.claude
        cp "$host_creds" /root/.claude/.credentials.json
        # Also copy settings if present
        [ -f "/root/.claude-host/settings.json" ] && cp "/root/.claude-host/settings.json" /root/.claude/settings.json
        echo "  claude: forwarded credentials from host (fallback copy)"
        return 0
    fi

    echo "  claude: no credentials found — run 'claude' inside the container to authenticate"
}

# Forward all host authentication into the container.
# Safe to call multiple times — only copies when host is newer or local is missing.
forward_host_auth() {
    echo "Forwarding host authentication..."
    forward_gh_auth
    forward_claude_auth
}

# Install a Claude Code plugin directly via git clone, bypassing the claude CLI.
# Usage: install_claude_plugin <git_repo_url> <plugin_name> <marketplace_name>
# Example: install_claude_plugin https://github.com/obra/superpowers.git superpowers claude-plugins-official
install_claude_plugin() {
    local repo_url="$1"
    local plugin_name="$2"
    local marketplace="${3:-claude-plugins-official}"
    local plugin_key="${plugin_name}@${marketplace}"

    local plugins_json="/root/.claude/plugins/installed_plugins.json"
    local cache_base="/root/.claude/plugins/cache/${marketplace}/${plugin_name}"

    # Check if already installed by looking at installed_plugins.json
    if [ -f "$plugins_json" ] && python3 -c "
import json, sys
d = json.load(open('$plugins_json'))
entries = d.get('plugins', {}).get('$plugin_key', [])
if entries and entries[0].get('installPath'):
    import os
    sys.exit(0 if os.path.isdir(entries[0]['installPath']) else 1)
sys.exit(1)
" 2>/dev/null; then
        echo "already installed"
        return 0
    fi

    # Clone the repo
    local tmpdir
    tmpdir=$(mktemp -d)
    if ! git clone --depth 1 "$repo_url" "$tmpdir" 2>/dev/null; then
        rm -rf "$tmpdir"
        return 1
    fi

    # Detect version: check git tags, then default to "1.0.0"
    local version sha
    sha=$(git -C "$tmpdir" rev-parse HEAD 2>/dev/null || echo "unknown")
    version=$(git -C "$tmpdir" describe --tags --exact-match HEAD 2>/dev/null || true)
    # Strip leading 'v' from version tags
    version="${version#v}"
    if [ -z "$version" ]; then
        version="1.0.0"
    fi

    local install_path="${cache_base}/${version}"
    mkdir -p "$(dirname "$install_path")"

    # Move the clone into the cache (preserves .git for future updates)
    rm -rf "$install_path"
    mv "$tmpdir" "$install_path"

    # Register in installed_plugins.json
    local now
    now=$(python3 -c "from datetime import datetime,timezone; print(datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.000Z'))")
    mkdir -p "$(dirname "$plugins_json")"
    python3 -c "
import json, os
path = '$plugins_json'
d = json.load(open(path)) if os.path.exists(path) else {'version': 2, 'plugins': {}}
d['plugins']['$plugin_key'] = [{
    'scope': 'user',
    'installPath': '$install_path',
    'version': '$version',
    'installedAt': '$now',
    'lastUpdated': '$now',
    'gitCommitSha': '$sha'
}]
json.dump(d, open(path, 'w'), indent=2)
"
    echo "installed v${version}"
}

# Ensure ORT CUDA provider .so files are installed in the system lib path.
# These can be lost when the ORT pyke.io cache is evicted (e.g., layer rebuild,
# driver update). When missing, corvia-inference silently falls back to CPU.
# Copies from the most recent cargo build if available.
ensure_ort_provider_libs() {
    # Only relevant on amd64 (CUDA/OpenVINO GPUs)
    if [ "$(dpkg --print-architecture)" != "amd64" ]; then
        return 0
    fi

    local ort_lib_dir="/usr/lib/x86_64-linux-gnu"
    local ort_libs="libonnxruntime_providers_shared libonnxruntime_providers_cuda libonnxruntime_providers_openvino"
    local libs_needed=false

    for lib in $ort_libs; do
        if [ ! -f "$ort_lib_dir/${lib}.so" ]; then
            libs_needed=true
            break
        fi
    done

    if [ "$libs_needed" = false ]; then
        return 0
    fi

    # Try release first, fall back to debug
    local target_dir="$WORKSPACE_ROOT/repos/corvia/target/release"
    if [ ! -d "$target_dir" ]; then
        target_dir="$WORKSPACE_ROOT/repos/corvia/target/debug"
    fi

    echo "    ORT provider libs missing — reinstalling from build..."
    local any_installed=false
    for lib in $ort_libs; do
        # Skip libs already present in destination
        [ -f "$ort_lib_dir/${lib}.so" ] && continue
        local src="$target_dir/${lib}.so"
        if [ -L "$src" ] || [ -f "$src" ]; then
            # Resolve symlinks (target/release/*.so → pyke.io cache); skip dangling
            local real_src
            real_src=$(readlink -f "$src" 2>/dev/null || true)
            if [ -n "$real_src" ] && [ -f "$real_src" ]; then
                sudo cp -L "$real_src" "$ort_lib_dir/${lib}.so"
                any_installed=true
            fi
        fi
    done

    if [ "$any_installed" = true ]; then
        sudo ldconfig
        echo "    ORT provider libs restored"
    else
        echo "    ORT provider libs not found in build — run 'corvia-dev rebuild' to generate"
    fi
}

# Ensure all tooling is installed (catches up if post-create was incomplete).
ensure_tooling() {
    ensure_corvia

    if ! command -v corvia-dev >/dev/null 2>&1; then
        echo "corvia-dev not found — installing..."
        retry 3 install_python_editable "$WORKSPACE_ROOT/tools/corvia-dev"
    fi
}
