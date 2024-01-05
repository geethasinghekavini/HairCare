import React from 'react';
import './Doctor.css'; 

const doctors = [
  {
    id: 1,
    firstName: 'DR(MRS).Shanika',
    lastName: 'Abeykeerthi',
    mobileNumber: '0117 888 888',
    email: 'shanikaabey93@gmail.com',
    hospital: 'Hemas Hospital Thalawathugoda',
    district: 'Colombo'
  },
  {
    id: 2,
    firstName: 'DR.Uwayse',
    lastName: 'Ahamed',
    mobileNumber: '0312 239 186',
    email: 'uwayseahamed45@gmail.com',
    hospital: 'Durdans Medical Center - Negombo',
    district: 'Colombo'
  },
  {
    id: 3,
    firstName: 'Dr.Janaka',
    lastName: 'Akarawita',
    mobileNumber: '0115 577 111',
    email: 'janakaakarawita76@gmail.com',
    hospital: 'Nawaloka Hospital',
    district: 'Colombo'
  },
  {
    id: 4,
    firstName: 'DR.Niranjan',
    lastName: 'Ariyasinghe',
    mobileNumber: '0115 430 000',
    email: 'niranjanariyasinghe57@gmail.com',
    hospital: 'Lanka Hospitals',
    district: 'Colombo'
  },
  {
    id: 5,
    firstName: 'DR(MRS).Dananja',
    lastName: 'Ariyawansa',
    mobileNumber: '0112 778 610',
    email: 'dananjaariyawansa34@gmail.com',
    hospital: 'Sri Jayewardenepura General Hospital',
    district: 'Colombo'
  },
  {
    id: 6,
    firstName: 'DR(MS).Damayanthi',
    lastName: 'Bandara',
    mobileNumber: '0115 577 111',
    email: 'damayanthibandara51@gmail.com',
    hospital: 'Nawaloka Hospital',
    district: 'Colombo'
  },
  {
    id: 7,
    firstName: 'DR(MRS).Manel',
    lastName: 'Dissanayake',
    mobileNumber: '0817 770 700',
    email: 'maneldissanayake37@gmail.com',
    hospital: 'KCC - Kandy',
    district: 'Kandy'
  },
  {
    id: 8,
    firstName: 'DR(MRS).Januka',
    lastName: 'Galahitiyawa',
    mobileNumber: '0115 430 000',
    email: 'janukagalahitiyawa98@gmail.com',
    hospital: 'Lanka Hospitals',
    district: 'Colombo'
  },
  {
    id: 9,
    firstName: 'DR.Saman',
    lastName: 'Gunasekara',
    mobileNumber: '0114 308 877',
    email: 'samangunasekara93@gmail.com',
    hospital: 'Ceymed Healthcare Services (Pvt) Ltd - Nugegoda.',
    district: 'Colombo'
  },
   
];

function Doctor() {
    return (
      <div className="doctor-page">
        {doctors.map(doctor => (
          <div key={doctor.id} className="doctor-card">
            <div className="doctor-details">
              <div className="doctor-name">
                {doctor.firstName} {doctor.lastName}
              </div>
              <div className="doctor-contact">
                <div className="doctor-mobile">{doctor.mobileNumber}</div>
                <div className="doctor-email">{doctor.email}</div>
              </div>
              <div className="doctor-hospital">{doctor.hospital}</div>
              <div className="doctor-district">{doctor.district}</div>
            </div>
          </div>
        ))}
      </div>
    );
  }
  
  export default Doctor;
