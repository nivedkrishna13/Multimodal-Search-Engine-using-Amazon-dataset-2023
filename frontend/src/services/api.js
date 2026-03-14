import axios from "axios";

const API = axios.create({
  baseURL: "http://127.0.0.1:8000"
});

export const searchText = (text) =>
  API.post("/search/text", {
    query:text 
  });

export const searchImage = (file) => {
  const formData = new FormData();
  formData.append("file", file);
  return API.post("/search/image", formData);
};

export const searchVoice = (file) => {
  const formData = new FormData();
  formData.append("file", file);
  return API.post("/search/voice", formData);
};