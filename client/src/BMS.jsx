import React, { useState } from "react";
import axios from "axios";

function BMS() {
  // State to hold the number input, file upload, and text input
  const [numOfSpeakers, setNumOfSpeakers] = useState(0);
  const [file, setFile] = useState(null);

  const [fileUploaded, setFileUploaded] = useState(false);
  const [uploadedFilePath, setUploadedFilePath] = useState("");

  const [summarizing, setSummarazing] = useState(false);
  const [summarized, setSummarized] = useState(""); // New state for the textarea

  const api = "http://127.0.0.1:5000";

  const handleNumOfSpeakersChange = (event) => {
    setNumOfSpeakers(event.target.value);
  };

  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
  };

  const handleUpload = async () => {
    if (!file) {
      alert("Please select a file first.");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await axios.post(`${api}/upload`, formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });

      setFileUploaded(true);
      setUploadedFilePath(response.data.file_path);
      alert(response.data.message);
    } catch (error) {
      console.error("Upload failed:", error);
      alert("Error uploading file.");
    }
  };

  const handleSummarize = async () => {
    if (!fileUploaded) {
      alert("Please upload a file to summarize.");
      return;
    }

    if (numOfSpeakers === 0) {
      alert("Please input the number of speakers.");
      return;
    }

    const formData = new FormData();
    formData.append("file_path", uploadedFilePath);
    formData.append("num_of_speakers", numOfSpeakers);

    try {
      setSummarazing(true);

      const response = await axios.post(`${api}/summarize`, formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });

      setSummarazing(false);
      setSummarized(response.data.summarized);
    } catch (error) {
      console.error("Summarization failed:", error);
      alert("Error during summarization.");
    }
  };

  return (
    <div className="flex flex-col h-full w-4/5 justify-center">
      <h1 className="text-5xl text-blue-500 m-2 font-bold">
        Business Meeting Summarization
      </h1>

      <div className="mb-4">
        <label className="text-xl text-blue-500 m-2 font-bold block">
          {"Number of speakers: "}
          <input
            className="bg-slate-600 text-xl p-1 font-semibold w-1/5 h-1/4 text-slate-300 "
            type="number"
            value={numOfSpeakers}
            onChange={handleNumOfSpeakersChange}
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
          accept="video/*"
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

      <label className="text-3xl text-blue-500 m-2 font-bold">Summary:</label>
      <textarea
        value={summarizing ? "Summarazing..." : summarized}
        placeholder="The summary of your meeting will appear here..."
        className="bg-slate-600 text-2xl p-2 font-semibold h-1/4 text-slate-300 resize-none"
        readOnly
      />
    </div>
  );
}

export default BMS;
