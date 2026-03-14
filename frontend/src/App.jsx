import React from "react";
import { useState } from "react";
import SearchBar from "./components/SearchBar";
import ProductCard from "./components/ProductCard";
import { searchText } from "./services/api";
import ImageUpload from "./components/ImageUpload";
import VoiceUpload from "./components/VoiceUpload";
function App() {
  const [results, setResults] = useState([]);

  const handleSearch = async (query) => {
  try {
    const res = await searchText(query);

    console.log(res.data); // 👈 see actual backend response

    setResults(res.data.results || []);
  } catch (err) {
    console.error("Search failed:", err);
    setResults([]);
  }
};

  return (
    <div style={{ padding: "40px" }}>
      <h1>Multimodal Search Engine 🔥</h1>
      <SearchBar onSearch={handleSearch} />
      <ImageUpload onResults={setResults} />
      <VoiceUpload onResults={setResults} />

      <div style={{
        display: "flex",
        flexWrap: "wrap"
      }}>
        {results.map((product, index) => (
          <ProductCard key={index} product={product} />
          
        ))}
      </div>
    </div>
  );
}

export default App;