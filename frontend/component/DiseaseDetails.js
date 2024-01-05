import React, { useState } from 'react';
import "./DiseaseDetails.css"; 
import Doctor from './Doctor';
 const DiseaseDetails = () => {
  //const [selectedDisease, setSelectedDisease] = useState(null);
  const [doctorDetailsClicked, setDoctorDetailsClicked] = useState(false);
  const handleDoctorDetailsClick = () => {
    setDoctorDetailsClicked(true);
  };
  // const handleDiseaseSelect = () => {
  //   setSelectedDisease(diseases[1]);
  // };
  // const handleFileSelect = (event) => {  
  //   setSelectedFile(event.target.files[0]);
  //   setPrediction(null);
  // };
  
  
  const diseases = [
    {
      name: 'Alopecia Areata',
      description: 'Alopecia areata is an autoimmune disorder that causes your hair to come out, often in clumps or in the shape of a quarter. The amount of hair loss is different for everyone.',
      image1: 'alopecia_areata_l.jpg',
      image2: 'alopecia_areata_s1.jpg',
      image3: 'alopecia_areata_s2.jpg',
      image4: 'alopecia_areata_s3.jpg',
      selfTreatment: 'Mix honey and a few drops of lemon and apply it on your scalp.Make a Fenugreek seeds paste and apply it on your scalp and leave it overnight.Squeeze onion juice and apply it on your scalp. Mix garlic juice and coconut oil, simmer in low flame and aplly it on your scalp. ',
    },
    {
      name: 'Contact Dermatitis',
      description: 'It is a condition where the hair scalp becomes itchy, red, dry, or blistering due to an allergy or an irritant which can be uncomfortable and painful sometimes.',
      image1: 'contact_dermatitis_l.jpg',
      image2: 'contact_dermatitis_s1.jpg',
      image3: 'contact_dermatitis_s2.jpg',
      image4: 'contact_dermatitis_s3.jpg',
      selfTreatment: '1.Avoid scratching your irritated skin.Clean your skin with mild soap and lukewarm water to remove any irritants.Apply petroleum jelly like Vaseline to soothe the area. Apply anti-itch treatments such as calamine lotion or hydrocortisone cream.',
    },
    {
      name: 'Folliculitis',
      description: 'Folliculitis is a common inflammatory skin condition affecting your hair follicles which can cause small, red bumps that look similar to an acne breakout. It’s usually caused by a fungal or bacterial infection.',
      image1: 'folliculitis_l.jpg',
      image2: 'folliculitis_s1.jpg',
      image3: 'folliculitis_s2.jpg',
      image4: 'folliculitis_s3.jpg',
      selfTreatment: '1.Apply warm, damp cloth to your scalp a few times a day. Wash your scalp with an anti-dandruff shampoo. Apply Cortisone cream or Antibiotic ointment. Wash your hair with anti-bacterial soap.',
    },
    {
      name: 'Head Lice',
      description: 'Head lice are tiny, crawling insects that live in the hair on your head. The most common symptom is itching, especially in the back of your head and neck and near your ears which has become common.',
      image1: 'head_lice_l.jpg',
      image2: 'head_lice_s1.jpg',
      image3: 'head_lice_s2.jpg',
      image4: 'head_lice_s3.jpg',
      selfTreatment: 'Wash your hair daily. Wet-combing your hair using a fine-toothed comb. Apply essential oil and leave it overnight. Mix garlic and few drops of lime juice and then apply the paste on your scalp.',
    },
    {
      name: 'Lichen Planus',
      description: 'Lichen Planus is a condition that causes scarring and patches of hair loss, usually on your scalp. It may also cause scalp pain, itching or burning.',
      image1: 'lichen_planus_l.jpg',
      image2: 'lichen_planus_s1.jpg',
      image3: 'lichen_planus_s2.jpg',
      image4: 'lichen_planus_s3.jpg',
      selfTreatment: ' Soak in an oatmeal bath. Avoid scratching. Apply cool compresses to a rash. Use OTC anti-itch creams.',
    },
    {
      name: 'Male pattern baldness',
      description: 'Male pattern baldness is hair loss and thinning of the hair that affects the hairline and top of the head. It is a very common condition for men over 40 years of age.',
      image1: 'male_pattern_baldness_l.jpg',
      image2: 'male_pattern_baldness_s1.jpg',
      image3: 'male_pattern_baldness_s2.jpg',
      image4: 'male_pattern_baldness_s3.jpg',
      selfTreatment: 'Apply Garlic, onion, ginger juice on your scalp. Apply a paste of a few drops of Apple Cider Vinegar with water on your scalp. Applying Essential oils on your scalp. ',
    },
    {
      name: 'Psoriasis',
      description: 'Psoriasis is an immune-mediated disease that causes a rash with itchy, scaly patches on the scalp.It can be itchy and uncomfortable. ',
      image1: 'psoriasis_l.jpg ',
      image2: 'psoriasis_s1.jpg ',
      image3: 'psoriasis_s2.jpg',
      image4: 'psoriasis_s3.jpg',
      selfTreatment: '1.Apply creams that contain 0.5% aloe vera. Apply a mixture of baking soda and water on the affected area. Having an Oatmeal bath. Use shampoo that contain 2% to 10% coal tar or anti-inflammatory salicylic acid.',
    },
    {
      name: 'Seborrheic dermatitis',
      description: 'Seborrheic dermatitis is a common, noncontagious, easy-to-manage skin condition. It causes white or yellow crusty or powdery flakes on your scalp, which can also be like dandruff.',
      image1: 'seborrheic_dermatitis_l.jpg',
      image2: 'seborrheic_dermatitis_s1.jpg',
      image3: 'seborrheic_dermatitis_s2.jpg',
      image4: 'seborrheic_dermatitis_s3.jpg',
      selfTreatment: 'Cant give self treatment methods. Please contact Dematologist',
    },
    {
      name: 'Telogen Effluvium ',
      description: 'Telogen effluvium is a common type of hair loss that affects people after they experience severe stress or a change to their body. Symptoms include thinning hair, usually around the top of your head. ',
      image1: 'telogen_effluvium_l.jpg ',
      image2: 'telogen_effluvium_s1.jpg ',
      image3: 'telogen_effluvium_s2.jpg ',
      image4: 'telogen_effluvium_s3.jpg ',
      selfTreatment: 'Cant give self treatment methods. Please contact Dematologist',
    },
    {
      name: 'Tinea capitis',
      description: 'Tinea capitis or scalp ringworm, is a fungal infection that affects your scalp and hair. Symptoms include swollen red patches, dry scaly rashes, itchiness and hair loss and it can be contagious.',
      image1: 'tinea_capitis_l.jpg',
      image2: 'tinea_capitis_s1.jpg',
      image3: 'tinea_capitis_s2.jpg',
      image4: 'tinea_capitis_s3.jpg',
      selfTreatment: 'Dab with Tea Tree Oil. Rinse with Apple Cider Vinegar.',
    }
  ];
  // const diseaseIndex = 1;
  // const handleDiseaseSelect = (diseaseIndex) => {
  //   setSelectedDisease(diseases[diseaseIndex]);
  // };
  // const handleDiseaseSelect = () => {
  //   setSelectedDisease(diseases[1]);
  // };
  //return (
    
      {/* <button onClick={handleDiseaseSelect} className="ghost">
        '''Successfully Predicted the Disease 
        Click Here to view details!!'''
       </button> */}
// function Disease() { 
//   const [showDoctor, setShowDoctor] = useState(false);
//   const toggleDoctor = () => {
//     setShowDoctor(!showDoctor);
//   };
//   const [prediction] = useState(null);
//   {prediction !== null ? (
//     <p className="prediction-text">Prediction: {prediction}</p>
//   ) : <> </>}
  
  return (
    <div>
    {doctorDetailsClicked ? (
      <Doctor />
    ) : (
      <div className="details">
        <div className="image-card1"><img src={diseases.image1} alt="image1" /></div>
        <div className="image-card2"><img src={diseases.image2} alt="image2" /></div>
        <div className="image-card3"><img src={diseases.image3} alt="image3" /></div>
        <div className="image-card4"><img src={diseases.image4} alt="image4" /></div>
        <div className="details-card1"><h2>{diseases.name}</h2></div>
        <div className="details-card2"><p>{diseases.description}</p></div>
        <div className="details-card3"><p>{diseases.selfTreatment}</p></div>
        <button onClick={handleDoctorDetailsClick}>View Doctor Details</button>
        <button className="doc-dets-button" onClick={handleDoctorDetailsClick}>View Doctor Details</button>
       </div>
        )} 
        </div>)
 }
export default DiseaseDetails;
