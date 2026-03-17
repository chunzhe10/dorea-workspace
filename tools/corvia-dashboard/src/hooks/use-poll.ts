import { useState, useEffect, useRef, useCallback } from "preact/hooks";

/**
 * Visibility-aware polling hook.
 *
 * Pauses polling when the page is not visible:
 * - **Browser**: uses `document.visibilitychange` (reliable in all modern browsers)
 * - **VS Code webview**: listens for `postMessage({ type: "visibility", visible })` from
 *   the extension host, since Electron's visibilitychange is unreliable (#28677).
 *
 * Resumes immediately with a fresh fetch when visibility is restored.
 */
export function usePoll<T>(
  fetcher: () => Promise<T>,
  intervalMs = 5000,
): { data: T | null; error: string | null; loading: boolean } {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const mounted = useRef(true);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const visibleRef = useRef(true);

  const poll = useCallback(async () => {
    if (!visibleRef.current) return;
    try {
      const result = await fetcher();
      if (mounted.current) {
        setData(result);
        setError(null);
        setLoading(false);
      }
    } catch (e: any) {
      if (mounted.current) {
        setError(e.message || "fetch failed");
        setLoading(false);
      }
    }
  }, [fetcher]);

  const startTimer = useCallback(() => {
    if (timerRef.current) clearInterval(timerRef.current);
    timerRef.current = setInterval(poll, intervalMs);
  }, [poll, intervalMs]);

  const stopTimer = useCallback(() => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  useEffect(() => {
    mounted.current = true;
    visibleRef.current = true;
    poll();
    startTimer();

    // Browser: pause when tab is hidden (works in standalone, NOT in VS Code webview)
    const onVisibilityChange = () => {
      if (document.hidden) {
        visibleRef.current = false;
        stopTimer();
      } else {
        visibleRef.current = true;
        poll();
        startTimer();
      }
    };
    document.addEventListener("visibilitychange", onVisibilityChange);

    // VS Code webview: extension host relays visibility via postMessage
    const onMessage = (e: MessageEvent) => {
      if (e.data && e.data.type === "visibility") {
        if (e.data.visible) {
          visibleRef.current = true;
          poll();
          startTimer();
        } else {
          visibleRef.current = false;
          stopTimer();
        }
      }
    };
    window.addEventListener("message", onMessage);

    return () => {
      mounted.current = false;
      stopTimer();
      document.removeEventListener("visibilitychange", onVisibilityChange);
      window.removeEventListener("message", onMessage);
    };
  }, [poll, intervalMs, startTimer, stopTimer]);

  return { data, error, loading };
}
