import { useState, useEffect, useCallback, useRef } from "react";
import { getHealth } from "../../api/health";
import { ApiError } from "../../api/types";
import "./SystemStatus.css";

type StatusState = "loading" | "online" | "degraded" | "offline";

const POLL_INTERVAL_MS = 30_000;

const STATUS_LABELS: Record<StatusState, string> = {
  loading: "Checking…",
  online: "System Online",
  degraded: "System Degraded",
  offline: "System Offline",
};

export function SystemStatus() {
  const [status, setStatus] = useState<StatusState>("loading");
  const mountedRef = useRef(true);

  const checkHealth = useCallback(async () => {
    try {
      await getHealth();
      if (mountedRef.current) setStatus("online");
    } catch (err) {
      if (!mountedRef.current) return;
      if (err instanceof ApiError && err.status === 503) {
        setStatus("degraded");
      } else {
        setStatus("offline");
      }
    }
  }, []);

  useEffect(() => {
    mountedRef.current = true;
    checkHealth();

    const id = setInterval(checkHealth, POLL_INTERVAL_MS);
    return () => {
      mountedRef.current = false;
      clearInterval(id);
    };
  }, [checkHealth]);

  return (
    <span className={`system-status system-status--${status}`}>
      <span className="system-status-dot" />
      {STATUS_LABELS[status]}
    </span>
  );
}
