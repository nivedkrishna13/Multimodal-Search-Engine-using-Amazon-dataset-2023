import { searchVoice } from "../services/api";
import React from "react";
export default function VoiceUpload({ onResults }) {

  const handleVoice = async (e) => {
    const file = e.target.files[0];
    const res = await searchVoice(file);
    onResults(res.data.results);
  };

  return (
    <div>
      <input type="file" accept="audio/*" onChange={handleVoice} />
    </div>
  );
}