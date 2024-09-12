import React, { useState } from 'react';
import classes from './ImproveModal.module.css';
import { improveTranslation } from '../../http/feedbackAPI';

const ImproveModal = (props) => {
  const [userTranslation, setUserTranslation] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    const data = {
        source_lng: props.sourceLng,
        target_lng: props.targetLng,
        source_txt: props.sourceText,
        translated_txt_our: props.translationText,
        translated_txt_user: userTranslation.trim(),
        page_lng: props.pageLanguage
    }
    improveTranslation(data)
    props.isOpened(false)
    props.setThanks(true)
  };

  return (
    <div className={classes["modal-improve"]}>
      <span class={classes["close-button"]} onClick={() => props.isOpened(false)}>✕</span>
      <div className={classes["modal-improve-content"]}>
        <div className={classes["modal-improve-title"]}>{props.textLayout[props.pageLanguage]['improve_trans']}</div>
        <form onSubmit={handleSubmit}>
          <label>{props.textLayout[props.pageLanguage]['source_text']}</label>
          <textarea readOnly value={props.sourceText} />
          <label>{props.textLayout[props.pageLanguage]['your_translation']}</label>
          <textarea 
            value={userTranslation}
            onChange={(e) => setUserTranslation(e.target.value)}
            placeholder={props.textLayout[props.pageLanguage]['enter_your_translation']}
          />
          <button type="submit" disabled={!userTranslation.trim()}>{props.textLayout[props.pageLanguage]['send']}</button>
        </form>
      </div>
    </div>
  );
};

export default ImproveModal;