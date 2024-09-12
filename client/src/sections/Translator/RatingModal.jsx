import React, { useState } from 'react';
import classes from './RatingModal.module.css'; // Добавь стили
import { rateTranslation } from '../../http/feedbackAPI';

const RatingModal = (props) => {
  const [selectedRating, setSelectedRating] = useState(0);
  const [isRated, setIsRated] = useState(false);

  const handleStarClick = (rating) => {
    setIsRated(true);
    const data = {
        source_lng: props.sourceLng,
        target_lng: props.targetLng,
        source_txt: props.sourceText,
        translated_txt: props.translationText,
        page_lng: props.pageLanguage,
        rating: rating
    }
    rateTranslation(data)
  };

  const handleCloseModal = () => {
    setSelectedRating(0); // Сбрасываем рейтинг
    setIsRated(false);
    props.setIsRating(false);
  };

  return (
    <div className={classes['rating-modal']} id={classes['ratingModal']}>
      {!isRated ? (
        <>
          <button 
            className={classes['close-button']}
            onClick={handleCloseModal}
            style={{
              position: 'absolute',
              top: '15px',
              right: '15px'
            }}
          >
            ✕
          </button>
          <div className={classes['modal-title']}>
            {props.textLayout[props.pageLanguage]['rate_trans_long']}
          </div>
          <div className={classes['stars']}>
            {[1, 2, 3, 4, 5].map((star) => (
              <button
                key={star}
                onClick={() => handleStarClick(star)}
                onMouseEnter={() => setSelectedRating(star)}
                className={star <= selectedRating ? classes['filled'] : ''}
              >
                ★
              </button>
            ))}
          </div>
        </>
      ) : (
        <div className={classes['thank-you']}>
          <p>{props.textLayout[props.pageLanguage]['thankyou-rate']}</p>
          <button className={classes['close-button']} onClick={handleCloseModal}>
            ✕
          </button>
        </div>
      )}
    </div>
  );
};

export default RatingModal;
