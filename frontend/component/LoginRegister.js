import React, { useState } from "react";
import "./LoginRegister.css";
import axios from "axios";
import UploadImage from "./UploadImage";

const LoginRegister = () => {
  const [addclass, setaddclass] = useState("");
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [loginEmail, setLoginEmail] = useState('');
  const [loginPassword, setLoginPassword] = useState('');
  const [isLoggedIn, setIsLoggedIn] = useState(false);

  const handleRegisterSubmit = (e) => {
    e.preventDefault();
    axios.post('http://localhost:3306/api/register', {
      name,
      email,
      password,
    })
    .then((response) => {
      console.log(response.data);
      // Reset the form inputs after successful registration
      setName('');
      setEmail('');
      setPassword('');
    })
    .catch((error) => {
      console.log(error.response.data);
    });
  }
  
  const handleLoginSubmit = (e) => {
    e.preventDefault();
    axios.post('http://localhost:3306/api/login', {
      email: loginEmail,
      password: loginPassword,
    })
    .then((response) => {
      console.log(response.data);
      setIsLoggedIn(true);
      // Reset the form inputs after successful login
      setLoginEmail('');
      setLoginPassword('');
    })
    .catch((error) => {
      console.log(error.response.data);
    });
  }
  
  if (isLoggedIn) {
    return <UploadImage />;
  }

  return (
    <div className={`container ${addclass}`} id="container">
      <div className="form-container  sign-up-container">
        <form method="POST" onSubmit={handleRegisterSubmit}>
          <h1>Create Account</h1>
          <input type="text" placeholder="NAME" value={name} onChange={(e) => setName(e.target.value)} />
          <input type="email" placeholder="EMAIL" value={email} onChange={(e) => setEmail(e.target.value)} />
          <input type="password" placeholder="PASSWORD" value={password} onChange={(e) => setPassword(e.target.value)} />
          <button type="submit" onClick={() => setaddclass("")}>REGISTER</button>
        </form>
      </div>
      <div className="form-container sign-in-container">
        <form method="POST" onSubmit={handleLoginSubmit}>
          <h1>Login</h1>
          <input type="email" placeholder="EMAIL" value={loginEmail} onChange={(e) => setLoginEmail(e.target.value)} />
          <input type="password" placeholder="PASSWORD" value={loginPassword} onChange={(e) => setLoginPassword(e.target.value)} />
          <h4><u>forgot password?</u></h4>
          <button type="submit">LOGIN</button>
        </form>
      </div>
      <div className="overlay-container">
        <div className="overlay">
          <div className="overlay-panel overlay-left">
            <button
              className="ghost"
              id="signIn"
              onClick={() => setaddclass("")}
            >
              GO TO LOGIN
            </button>
          </div>
          <div className="overlay-panel overlay-right">
            <button
              className="ghost"
              id="signUp"
              onClick={() => setaddclass("right-panel-active")}
            >
              GO TO REGISTER
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default LoginRegister;