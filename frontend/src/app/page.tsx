"use client";
import { useState } from "react";

export default function Home() {
  const [uri, setUri] = useState("");
  const [pred, setPred] = useState<any>(null);

  const predict = async () => {
    const q = new URLSearchParams({ uri }).toString();
    const res = await fetch(`http://127.0.0.1:8000/predict/spotify_track?${q}`, { method: "POST" });
    setPred(await res.json());
  };

  return (
    <main className="mx-auto max-w-2xl p-6">
      <h1 className="text-3xl font-bold mb-4">Song Success Predictor</h1>
      <input
        placeholder="spotify:track:... or https://open.spotify.com/track/..."
        className="border rounded p-2 w-full mb-3"
        value={uri}
        onChange={(e) => setUri(e.target.value)}
      />
      <button className="rounded bg-black text-white px-4 py-2" onClick={predict}>Predict</button>
      {pred && (
        <pre className="mt-4 bg-gray-100 p-3 rounded text-sm overflow-x-auto">
{JSON.stringify(pred, null, 2)}
        </pre>
      )}
    </main>
  );
}
