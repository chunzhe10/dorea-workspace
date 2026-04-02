#!/bin/bash
# Runs on the HOST before the container is created.
# Detects GPU availability and generates docker-compose.override.yml.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DC_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
OVERRIDE="$DC_DIR/docker-compose.override.yml"

# If the .devcontainer dir or override file aren't writable (e.g. owned by
# root from a previous Docker rebuild), fix ownership so we can write to them.
if [ ! -w "$DC_DIR" ]; then
    sudo chown -R "$(id -u):$(id -g)" "$DC_DIR"
fi
if [ -e "$OVERRIDE" ] && [ ! -w "$OVERRIDE" ]; then
    rm -f "$OVERRIDE" 2>/dev/null || sudo rm -f "$OVERRIDE"
fi

HOME_DIR="${HOME:-${USERPROFILE:-}}"
if [ -z "$HOME_DIR" ]; then
    echo "Warning: Cannot determine home directory — auth forwarding may not work"
fi

# Create directories if they don't exist so Docker bind mounts succeed.
[ -n "$HOME_DIR" ] && mkdir -p "$HOME_DIR/.config/gh" "$HOME_DIR/.claude"

# ── Remove stale containers ─────────────────────────────────────────
# VS Code uses `docker compose up --no-recreate` which reuses existing
# containers even when the compose config has changed (e.g. tty, mounts).
# Remove any exited containers for this workspace so they get recreated
# with the current config.
WORKSPACE_DIR="$(cd "$DC_DIR/.." && pwd)"
STALE_IDS=$(docker ps -q -a \
    --filter "label=devcontainer.local_folder=$WORKSPACE_DIR" \
    --filter "status=exited" 2>/dev/null || true)
if [ -n "$STALE_IDS" ]; then
    echo "Removing exited devcontainer(s) to ensure fresh config..."
    echo "$STALE_IDS" | xargs docker rm 2>/dev/null || true
fi

# ── Platform detection ───────────────────────────────────────────────
IS_WSL=false
if grep -qi "microsoft\|wsl" /proc/version 2>/dev/null; then
    IS_WSL=true
    echo "Platform: WSL"
elif [ "$(uname -s)" = "Linux" ]; then
    echo "Platform: native Linux"
else
    echo "Platform: $(uname -s)"
fi

# ── GPU detection ────────────────────────────────────────────────────
HAS_NVIDIA=false
HAS_DRI=false
HAS_DXG=false

# NVIDIA: check for nvidia-smi and nvidia-container-toolkit
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    if command -v nvidia-container-cli >/dev/null 2>&1 || \
       [ -f /usr/bin/nvidia-container-runtime ]; then
        HAS_NVIDIA=true
        # Pre-initialize nvidia-uvm on the host. Without this, /dev/nvidia-uvm
        # is present inside the container but returns I/O error, causing cuInit
        # to fail with error 999 (CUDA_ERROR_UNKNOWN).  Initialization is a
        # one-time kernel-side operation that persists until reboot.
        if command -v nvidia-modprobe >/dev/null 2>&1; then
            sudo nvidia-modprobe -u -c=0 >/dev/null 2>&1 || \
                echo "GPU: Warning — nvidia-modprobe -u -c=0 failed (CUDA may not work inside container)"
        else
            sudo modprobe nvidia-uvm >/dev/null 2>&1 || \
                echo "GPU: Warning — modprobe nvidia-uvm failed (CUDA may not work inside container)"
        fi
        # Ensure CDI spec exists and matches the running driver version.
        # Docker uses CDI to discover GPUs; a missing or stale spec causes
        # "could not select device driver nvidia" even when the driver works.
        if command -v nvidia-ctk >/dev/null 2>&1; then
            RUNNING_DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || true)
            # If grep pattern fails (CDI format change), CDI_DRIVER will be empty
            # and we'll regenerate — safe, since generation is idempotent.
            CDI_DRIVER=$(grep -oP 'host-driver-version=\K[0-9.]+' /etc/cdi/nvidia.yaml 2>/dev/null | head -1 || true)
            if [ -z "$RUNNING_DRIVER" ]; then
                echo "GPU: Warning — could not determine driver version, skipping CDI check"
            elif [ ! -f /etc/cdi/nvidia.yaml ] || [ "$RUNNING_DRIVER" != "$CDI_DRIVER" ]; then
                echo "GPU: Regenerating NVIDIA CDI spec (driver: $RUNNING_DRIVER)..."
                if ! sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml >/dev/null 2>&1; then
                    echo "GPU: Warning — CDI spec regeneration failed (non-fatal)"
                fi
            fi
        else
            echo "GPU: nvidia-ctk not found — CDI spec may be stale after driver changes"
            echo "     Install nvidia-container-toolkit for automatic CDI management"
        fi
    else
        echo "GPU: NVIDIA GPU found but nvidia-container-toolkit not installed"
        echo "     Install it for GPU passthrough: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/"
        echo "     Ollama will run CPU-only until the toolkit is installed."
    fi
fi

# DRI render nodes (/dev/dri) — used by Intel, AMD, and sometimes NVIDIA
if [ -d /dev/dri ] && ls /dev/dri/renderD* >/dev/null 2>&1; then
    HAS_DRI=true
fi

# WSL2 DirectX GPU passthrough
if [ "$IS_WSL" = true ] && [ -e /dev/dxg ]; then
    HAS_DXG=true
fi

# ── Generate docker-compose.override.yml ─────────────────────────────
{
    echo "# Auto-generated by init-host.sh — do not edit manually."

    if [ "$HAS_NVIDIA" = false ] && [ "$HAS_DRI" = false ] && [ "$HAS_DXG" = false ]; then
        echo "# No GPU detected — running CPU-only."
        echo "services:"
        echo "  app: {}"
    else
        echo "# Detected: nvidia=$HAS_NVIDIA dri=$HAS_DRI wsl_dxg=$HAS_DXG"
        echo "services:"
        echo "  app:"

        # Collect devices
        DEVICES=()
        GROUP_ADD=()

        if [ "$HAS_DRI" = true ]; then
            DEVICES+=("/dev/dri:/dev/dri")
            # Use numeric GIDs — group names may not exist inside the container.
            VIDEO_GID=$(getent group video 2>/dev/null | cut -d: -f3 || true)
            RENDER_GID=$(getent group render 2>/dev/null | cut -d: -f3 || true)
            [ -n "$VIDEO_GID" ] && GROUP_ADD+=("$VIDEO_GID")
            [ -n "$RENDER_GID" ] && GROUP_ADD+=("$RENDER_GID")
        fi

        if [ "$HAS_DXG" = true ]; then
            DEVICES+=("/dev/dxg:/dev/dxg")
        fi

        # Device passthrough
        if [ ${#DEVICES[@]} -gt 0 ]; then
            echo "    devices:"
            for d in "${DEVICES[@]}"; do
                echo "      - $d"
            done
        fi

        # Group membership for DRI access
        if [ ${#GROUP_ADD[@]} -gt 0 ]; then
            echo "    group_add:"
            for g in "${GROUP_ADD[@]}"; do
                echo "      - $g"
            done
        fi

        # CAP_PERFMON for intel_gpu_top GPU monitoring (i915 PMU)
        if [ "$HAS_DRI" = true ]; then
            echo "    cap_add:"
            echo "      - PERFMON"
        fi

        # Intel GPU compute firmware (required for OpenCL/Level Zero)
        if [ -d /lib/firmware/i915 ]; then
            echo "    volumes:"
            echo "      - /lib/firmware/i915:/lib/firmware/i915:ro"
        fi

        # WSL2 driver libs
        if [ "$IS_WSL" = true ] && [ -d /usr/lib/wsl/lib ]; then
            echo "    volumes:"
            echo "      - /usr/lib/wsl:/usr/lib/wsl:ro"
        fi

        # NVIDIA container toolkit (uses deploy.resources for compose v2)
        if [ "$HAS_NVIDIA" = true ]; then
            echo "    deploy:"
            echo "      resources:"
            echo "        reservations:"
            echo "          devices:"
            echo "            - driver: nvidia"
            echo "              count: all"
            echo "              capabilities: [gpu]"
        fi
    fi
} > "$OVERRIDE"

# Print summary
echo "GPU: $(head -2 "$OVERRIDE" | tail -1 | sed 's/^# //')"
echo "Host init complete."
