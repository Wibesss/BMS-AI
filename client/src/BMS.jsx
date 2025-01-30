import React, { useState } from "react";

function BMS() {
  // State to hold the number input, file upload, and text input
  const [number, setNumber] = useState(0);
  const [file, setFile] = useState(null);
  const [text, setText] = useState(""); // New state for the textarea

  const api = "http://127.0.0.1:5000"

  // Handle number input change
  const handleNumberChange = (event) => {
    setNumber(event.target.value);
  };

  // Handle file input change
  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
  };

  // Handle text input change (textarea)
  const handleTextChange = (event) => {
    setText(event.target.value);
  };

  // Handle file upload (simulated)
  const handleUpload = () => {
    if (!file) {
      alert("Please upload a file.");
      return;
    }
    alert(`File ${file.name} uploaded successfully!`);
    // Add logic to handle actual file upload (e.g., using FormData and an API)
  };

  // Handle file summarize (simulated)
  const handleSummarize = () => {
    if (!file) {
      alert("Please upload a file to summarize.");
      return;
    }
    alert(
      `Summarizing file ${file.name} with number ${number} and text: ${text}`
    );
    // Add logic for actual summarization here
  };

  return (
    <div className="flex flex-col h-full w-4/5 justify-center">
      <h1 className="text-5xl text-blue-500 m-2 font-bold">
        Business Meeting Summarization
      </h1>

      <div className="mb-4">
        <label className="text-xl text-blue-500 m-2 font-bold block">
          {"Number of Participants: "}
          <input
            className="bg-slate-600 text-xl p-1 font-semibold w-1/5 h-1/4 text-slate-300 "
            type="number"
            value={number}
            onChange={handleNumberChange}
          />
        </label>
      </div>

      <div className="mb-4 flex ">
        <div className="flex items-center">
          <label className="text-xl text-blue-500 m-2 font-bold block">
            Upload File:
          </label>
        </div>
        <input
          type="file"
          accept="video/*" // Only allow video files
          onChange={handleFileChange}
          className="hidden"
          id="fileUpload"
        />
        <label htmlFor="fileUpload" className="btn-prim">
          {file ? file.name : "Choose File"}
        </label>
      </div>

      <div className="flex justify-around w-full mb-4">
        <button className="btn-prim" onClick={handleUpload}>
          Upload
        </button>
        <button className="btn-prim" onClick={handleSummarize}>
          Summarize
        </button>
      </div>

      {/* Textarea */}
      <label className="text-3xl text-blue-500 m-2 font-bold">Summary:</label>
      <textarea
        value={text}
        onChange={handleTextChange}
        placeholder="The summary of your meeting will appear here..."
        className="bg-slate-600 text-2xl p-2 font-semibold h-1/4 text-slate-300 resize-none"
      />
    </div>
  );
}

export default BMS;
