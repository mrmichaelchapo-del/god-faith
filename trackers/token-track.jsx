import React from "react";

const MAX_TOKENS = 20_000;

export default function TokenTracker({ tokens = 0 }) {
  const percentage = Math.min((tokens / MAX_TOKENS) * 100, 100);

  return (
    <div style={{ width: "100%", maxWidth: 500, fontFamily: "sans-serif" }}>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          marginBottom: 8,
        }}
      >
        <span>Token usage</span>
        <span>
          {tokens.toLocaleString()} / {MAX_TOKENS.toLocaleString()}
        </span>
      </div>

      <div
        style={{
          width: "100%",
          height: 12,
          background: "#ddd",
          borderRadius: 999,
          overflow: "hidden",
        }}
      >
        <div
          style={{
            width: `${percentage}%`,
            height: "100%",
            background: "#333",
            borderRadius: 999,
            transition: "width 0.2s ease",
          }}
        />
      </div>

      <div style={{ marginTop: 6, fontSize: 13 }}>
        {percentage.toFixed(1)}% used
      </div>
    </div>
  );
}