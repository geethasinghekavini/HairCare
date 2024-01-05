import React from 'react';
import student1 from './Images/linara.jpg';
import student2 from './Images/kavini2.jpg';
import student3 from './Images/dewmini.jpg';
import student4 from './Images/mehara.jpg';
import student5 from './Images/ishan.jpg';
import './AboutUs.css';

const AboutUs = () => {
  const students = [   
    {      name: 'Linara Wataraka',      uow: 'UoW ID - w1898948',      iit: 'IIT ID -20211343',       image: student1    },  
    {      name: 'Kavini Geethasinghe',      uow: 'UoW ID - w1867661',      iit: 'IIT ID - 20210545',      image: student2    },  
    {      name: 'Dewmini Ruwanpathirana',      uow: 'UoW ID - w1898925',      iit: 'IIT ID - 20211282',      image: student3    },    
    {      name: 'Mehara Weeratunga',      uow: 'UoW ID - w1914620',      iit: 'IIT ID - 20210301',      image: student4    },    
    {      name: 'Ishan Akmal Samsudeen',      uow: 'UoW ID - w1898937',      iit: 'IIT ID - 20211320',      image: student5    }  ];

  return (
    <div className="about-container" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
      <h1 style={{ fontSize: '56px', color: '#7b512e' }}>About Us</h1>
      <h4>The Software Engineering undergraduates who developed the ScalpCare prototype</h4>
      <div style={{ display: 'flex', flexWrap: 'wrap', justifyContent: 'center' }}>
        {students.map((student, index) => (
          <div key={index} className="about-student" style={{ width: '50%', maxWidth: '600px', margin: '20px', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
            <img src={student.image} alt={student.name} style={{ maxWidth: '200px', height: 'auto', borderRadius: '10px' }} />
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', marginTop: '10px' }}>
              <h2>{student.name}</h2>
              <p>{student.uow}</p>
              <p>{student.iit}</p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default AboutUs;
