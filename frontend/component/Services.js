import React from "react";
import './Service.css';

function Services ()  {
    return (
        <div className="service component__space" id="Services">
            <div className="heading">
                <h1 className="heading">Services Provided</h1>
                <p className="heading p__color">
                Below are some of the services that is been provided by the application.
                </p>
            </div>

            <div className="container">
                <div className="row">
                      <div className="service__box pointer">
                            <div className="icon">
                            <svg
                            stroke="currentColor"
                            fill="none"
                            stroke-width="2"
                            viewBox="0 0 24 24"
                            stroke-linecap="round"
                            stroke-linejoin="round"
                            height="1em"
                            width="1em"
                            xmlns="http://www.w3.org/2000/svg"
                          >          
                  <polygon points="12 2 2 7 12 12 22 7 12 2"></polygon>
                  <polyline points="2 17 12 22 22 17"></polyline>
                  <polyline points="2 12 12 17 22 12"></polyline>
                </svg>
                            </div>
                            <div className="service__meta">
                                <h1 className="service__text">Identifying the disease</h1>
                                <p className="p service__text p__color">
                                    A great way to identify your hair and scalp disease efficiently.
                                </p>
                                <p className="p service__text p__color">
                                    Just have to upload an image of your hair/scalp.
                                </p>
                            </div>
                         </div>                

                    
                         <div className="service__box pointer">
                            <div className="icon">
                            <svg
                            stroke="currentColor"
                            fill="none"
                            stroke-width="2"
                            viewBox="0 0 24 24"
                            stroke-linecap="round"
                            stroke-linejoin="round"
                            height="1em"
                            width="1em"
                            xmlns="http://www.w3.org/2000/svg"
                          >
                  
                  <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"></path>
                  <circle cx="9" cy="7" r="4"></circle>
                  <path d="M23 21v-2a4 4 0 0 0-3-3.87"></path>
                  <path d="M16 3.13a4 4 0 0 1 0 7.75"></path>
                </svg>
                            </div>
                            <div className="service__meta">
                                <h1 className="service__text">Recommending dermatologists</h1>
                                <p className="p service__text p__color">
                                  If you prefer to get in touch with a dermatologist the application is able to recommend dermatologists to you.
                                </p>
                                <p className="p service__text p__color">
                                  Provides well known dermatologists around hospitals in Srilanka.
                                </p>
                            </div>
                         </div>
                    

                 
                         <div className="service__box pointer">
                            <div className="icon">
                            <svg
                  stroke="currentColor"
                  fill="none"
                  stroke-width="2"
                  viewBox="0 0 24 24"
                  stroke-linecap="round"
                  stroke-linejoin="round"
                  height="1em"
                  width="1em"
                  xmlns="http://www.w3.org/2000/svg"
                >
                  <rect x="2" y="3" width="20" height="14" rx="2" ry="2"></rect>
                  <line x1="8" y1="21" x2="16" y2="21"></line>
                  <line x1="12" y1="17" x2="12" y2="21"></line>
                </svg>
                            </div>
                            <div className="service__meta">
                                <h1 className="service__text">Providing self treatment methods</h1>
                                <p className="p service__text p__color">
                                   This application provides self treatment methods for your hair and scalp diseases within a short period of time.
                                </p>
                                <p className="p service__text p__color">
                                    Save your money and time.
                                </p>
                            </div>
                         </div>
                </div>
            </div>
        </div>
    )
}

export default Services;
