const vscode = require("vscode");

let statusBarItem;
let browserBarItem;
let panel;
let pollTimer;

const POLL_INTERVAL = 5000;
const API_BASE = "http://localhost:8020";
const DASHBOARD_URL = "http://localhost:8021";

function activate(context) {
    statusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 50);
    statusBarItem.command = "corvia.openDashboard";
    statusBarItem.text = "$(loading~spin) Corvia";
    statusBarItem.show();
    context.subscriptions.push(statusBarItem);

    browserBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 49);
    browserBarItem.command = "corvia.openInBrowser";
    browserBarItem.text = "$(link-external)";
    browserBarItem.tooltip = "Open Corvia Dashboard in browser";
    browserBarItem.show();
    context.subscriptions.push(browserBarItem);

    context.subscriptions.push(
        vscode.commands.registerCommand("corvia.openDashboard", () => openDashboard(context)),
        vscode.commands.registerCommand("corvia.openInBrowser", () => {
            vscode.env.openExternal(vscode.Uri.parse(DASHBOARD_URL));
        })
    );

    pollStatus();
    pollTimer = setInterval(pollStatus, POLL_INTERVAL);
    context.subscriptions.push({ dispose: () => clearInterval(pollTimer) });
}

async function pollStatus() {
    try {
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), 3000);
        const resp = await fetch(`${API_BASE}/api/dashboard/status`, { signal: controller.signal });
        clearTimeout(timeout);
        const data = await resp.json();
        const allHealthy = (data.services || []).every(s => s.state === "healthy");
        const anyDown = (data.services || []).some(s => s.state !== "healthy");

        if (allHealthy) {
            statusBarItem.text = "$(check) Corvia";
            statusBarItem.backgroundColor = undefined;
        } else if (anyDown) {
            statusBarItem.text = "$(error) Corvia";
            statusBarItem.backgroundColor = new vscode.ThemeColor("statusBarItem.errorBackground");
        } else {
            statusBarItem.text = "$(warning) Corvia";
            statusBarItem.backgroundColor = new vscode.ThemeColor("statusBarItem.warningBackground");
        }

        const svcSummary = (data.services || [])
            .map(s => `${s.name}: ${s.state}`)
            .join(" | ");
        statusBarItem.tooltip = svcSummary;
    } catch {
        statusBarItem.text = "$(error) Corvia";
        statusBarItem.backgroundColor = new vscode.ThemeColor("statusBarItem.errorBackground");
        statusBarItem.tooltip = "corvia-server not responding";
    }
}

function openDashboard(context) {
    if (panel) {
        panel.reveal();
        return;
    }

    panel = vscode.window.createWebviewPanel(
        "corviaDashboard",
        "Corvia Dashboard",
        vscode.ViewColumn.One,
        { enableScripts: true, retainContextWhenHidden: true }
    );

    panel.webview.html = getWebviewContent();

    // Relay visibility state to the dashboard iframe so it can pause/resume polling.
    // document.visibilitychange is unreliable in Electron webviews (electron#28677),
    // so we use the VS Code API instead.
    panel.onDidChangeViewState(e => {
        try {
            panel.webview.postMessage({
                type: "visibility",
                visible: e.webviewPanel.visible,
            });
        } catch { /* panel may be disposed */ }
    });

    panel.onDidDispose(() => { panel = undefined; }, null, context.subscriptions);
}

function getWebviewContent() {
    return `<!DOCTYPE html>
<html>
<head>
  <meta http-equiv="Content-Security-Policy"
        content="default-src 'none'; frame-src ${DASHBOARD_URL}; style-src 'unsafe-inline'; script-src 'unsafe-inline';" />
  <style>
    body, html { margin: 0; padding: 0; width: 100%; height: 100%; overflow: hidden; }
    iframe { width: 100%; height: 100%; border: none; }
  </style>
</head>
<body>
  <iframe id="dash" src="${DASHBOARD_URL}" />
  <script>
    // Relay VS Code visibility events to the dashboard iframe
    window.addEventListener("message", (e) => {
      if (e.data && e.data.type === "visibility") {
        document.getElementById("dash").contentWindow.postMessage(e.data, "*");
      }
    });
  </script>
</body>
</html>`;
}

function deactivate() {
    if (pollTimer) clearInterval(pollTimer);
}

module.exports = { activate, deactivate };
