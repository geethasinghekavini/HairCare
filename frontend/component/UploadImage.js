import React, { useState } from "react";     
import "./UploadImage.css";                   // importing the CSS file.
import upload from './Images/upload.png';     // import image for upload button
import axios from 'axios';                    // import axios library for HTTP requests
import DiseaseDetails from './DiseaseDetails';

// Creating a new function called UploadImage.
const UploadImage = () => {   
  const [diseaseDetailsClicked, setDiseaseDetailsClicked] = useState(false);
  const handleDeseaseDetailsClick = () => {
    setDiseaseDetailsClicked(true);
  };               
  const [selectedFile, setSelectedFile] = useState(null);   
  const [prediction, setPrediction] = useState(null);

  // function to handle file selection.
  const handleFileSelect = (event) => {  
    setSelectedFile(event.target.files[0]);
    setPrediction(null);
  };
  
  // function to handle form submission
  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!selectedFile) return;
  
    // create form data object
    const formData = new FormData();
    console.log(selectedFile)
    formData.append("image", selectedFile);
  
    try {
      // const response = await axios.post("http://172.27.24.181:5000/predict", formData);
      // send POST request to server
      const response = await axios({      
        method:"post",
        url:"http://127.0.0.1:5000/predict",
        data: formData,
        headers: {'Content-Type': 'multipart/form-data', 'Content-Disposition': 'attachment'}
      }).then((res) =>{return res});

      console.log(response.data)       // log prediction result to console
      setPrediction(response.data);    // update state with prediction result
    } catch (error) {
      console.error(error);            // log any errors to console
    }
  };
  
  
  return (
    <div>
    {diseaseDetailsClicked ? (
      <DiseaseDetails />
    ) : (
    <div className="card">
      <div className="card-header">
        <h5 className="card-title">Image Upload</h5>
      </div>
      <div className="card-body">
        {selectedFile ? (
          <>
            <img
              src={URL.createObjectURL(selectedFile)}
              alt="Uploaded File"
              style={{ maxWidth: "100%", marginBottom: "10px" }}
            />
            {prediction !== null ? (
              <p className="prediction-text">Prediction: {prediction}</p>
            ) : (
              <button className="predict-button" onClick={handleSubmit}>Predict</button>
            )}
          </>
        ) : (
          <img
            src={upload}
            className="upload-image"
            alt="upload image"
          />
        )}
        <button
          className="upload-image-button"
          onClick={() => document.getElementById("filePicker").click()}
        >
          Upload Image
        </button>
        <input
          id="filePicker"
          type="file"
          accept="image/*"
          onChange={handleFileSelect}
          style={{ display: "none" }}
        />
      </div>
      <button className="View-disease-details-button" onClick={handleDeseaseDetailsClick}>View Disease Details</button>
    </div>
  )
}
</div>)}

export default UploadImage;
