import { searchImage } from "../services/api";
import React from "react";

export default function ImageUpload({ onResults }) {

  const handleImage = async (e) => {
    const file = e.target.files[0];
    const res = await searchImage(file);
    onResults(res.data.results);
  };

  return (
    <div>
      <input type="file" accept="image/*" onChange={handleImage} />
    </div>
  );
}