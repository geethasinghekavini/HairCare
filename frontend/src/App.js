import logo from './logo.png';
import home from './Images/home.png';
import './App.css';
import React, { useState } from 'react';
import LoginRegister from './Components/LoginRegister';
import Services from './Components/Services';

function App() {
  const [showHome, setShowHome] = useState(false);
  const [showLogin, setShowLogin] = useState(false);
  const [showServices, setShowServices] = useState(false);

  const toggleHome = () => {
    setShowHome(!showHome);
  };
  const toggleLogin = () => {
    setShowLogin(!showLogin);
  };
  const toggleServices = () => {
    setShowServices(!showServices);
  };

  return (
    <div className={`root`}>
      <header className="header">
        <img src={logo} alt="Skin Disease Detection logo" className="header-logo"/>
        <nav className="header-nav">
          <ul className="header-nav-list">
            <li className="header-nav-item">
              <a href="#" className="header-nav-link" onClick={toggleHome}>HOME</a>
            </li>
            <li className="header-nav-item">
              <a href="#" className="header-nav-link">ABOUT US</a>
            </li>
            <li className="header-nav-item">
              <a href="#" className="header-nav-link" onClick={toggleServices}>SERVICES</a>
            </li>
            <li className="header-nav-item">
              <a href="#" className="header-nav-link">CONTACT US</a>
            </li>
          </ul>
        </nav>
        <button className="header-login-button" onClick={toggleLogin}>
          Login
        </button>
      </header>
      <main className="main">
        {!showLogin && !showServices && (
          <>
            <div className='right-cont'>
              <h1 className='header-title'> ScalpCare </h1>
              <h4 className='head-text2'>HAIR AND SCALP DISEASE IDENTIFIER </h4>
              <button className="right-cont.detect-button" onClick={toggleLogin}>
                Get Started
              </button>
            </div>
            <div className='left-cont'>
              <img src={home} className='home-image' alt='home image'></img>
            </div>
          </>
        )}
      </main>
      {showHome && (
        <div className="login-wrapper">
          <App />
        </div>
      )}
      {showLogin && (
        <div className="login-wrapper">
          <LoginRegister />
        </div>
      )}
      {showServices && (
        <div className="login-wrapper">
          <Services />
        </div>
      )}
    </div>
  );
}

export default App;
