import React from "react";


export default function ProductCard({ product }) {
  return (
    <div style={{
      border: "1px solid #ddd",
      borderRadius: "8px",
      padding: "10px",
      margin: "10px",
      width: "250px",
      boxShadow: "0px 2px 5px rgba(0,0,0,0.1)"
    }}>
      <img
        src={product.image}
        alt={product.title}
        style={{ width: "100%", height: "200px", objectFit: "cover" }}
      />
      <h4>{product.title}</h4>
      <p>💲 {product.price}</p>
      <p>⭐ {product.stars}</p>
      <p>{product.category}</p>
    </div>
  );
}