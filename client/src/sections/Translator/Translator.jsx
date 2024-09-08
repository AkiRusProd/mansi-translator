// eslint-disable-next-line
import React, { useState, useEffect, useRef} from 'react';
import classes from './Translator.module.css';
import { translate } from '../../http/translationAPI';
import Loading from '../Loading';

const addSymb = [
  'а̄', 'ē', 'ё̄', 'ӣ', 'ӈ', 'о̄', 'ӯ', 'ы̄', 'э̄', 'ю̄', 'я̄'
]

// function sleep(ms) {
//   return new Promise(resolve => setTimeout(resolve, ms));
// }

const Translator = () => {
  const [sourceLng, setSourceLng] = useState('Русский')
  const [targetLng, setTargetLng] = useState('Мансийский')
  const [translationText, setTranslationText] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [sourceText, setSourceText] = useState('')
  const typingTimeoutRef = useRef(null)

  const switchLanguages = () => {
    let src = sourceLng.slice()
    let target = targetLng.slice()
    setSourceLng(target)
    setTargetLng(src)
    if (translationText.length > 0) {
      setIsLoading(true)
      let src_text = translationText.slice()
      sendTranslation(src_text)
    }
  }

  const clearIO = () => {
    setIsLoading(false)
    setSourceText('')
    setTranslationText('')
  }

  const handleInputChange = (e) => {
    if (e.target.value.trim().length === 0) {
      clearIO()
      return
    }
    setSourceText(e.target.value)
    setIsLoading(true)
    if (typingTimeoutRef.current) {
      clearTimeout(typingTimeoutRef.current)
    }
    typingTimeoutRef.current = setTimeout(() => {
      sendTranslation(e.target.value)
    }, 1000)
  }

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      if (typingTimeoutRef.current) {
        clearTimeout(typingTimeoutRef.current)
      }
      sendTranslation(sourceText)
    }
  }

  const sendTranslation = async (text) => {
    try {
      // await sleep(200)
      if (text.trim().length === 0) {
        clearIO()
        return
      }
      const data = {
        text: text.trim(),
        source_lang: (sourceLng === 'Мансийский') ? ('mansi_Cyrl') : ('rus_Cyrl'),
        target_lang: (targetLng === 'Мансийский') ? ('mansi_Cyrl') : ('rus_Cyrl')
      }
      const translation = await translate(data);
      //const translation = {translated_text: 'new text' + text}
      setTranslationText(translation.translated_text)
    } catch (error) {
      console.error('Error translating text:', error)
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <section id={classes.translator}>
      <h3>Русско-мансийский и мансийско-русский переводчик</h3>
      <div className={classes.main}>
        <div className={classes.switch}>
          <div className={classes.lng_name}>{sourceLng}</div>
          <div className={classes.switch_icon_holder} onClick={switchLanguages}>⇄</div>
          <div className={classes.lng_name}>{targetLng}</div>
        </div>
        <div className={classes.input_area}>
          <div className={classes.source}>
          {
            sourceLng === 'Мансийский' ? (
            <div className={classes.addition_symbols}>
              <p>
                {addSymb.map((symbol, index) => (
                  <span key={index} onClick={() => setSourceText(sourceText.slice() + symbol)}>{symbol}</span>
                ))}
              </p>
            </div>
            ) : ('')
          }
            
          <textarea
              onChange={handleInputChange}
              onKeyDown={handleKeyPress}
              value={sourceText}
            />
          </div>
          <div className={classes.cancel}>
            <button 
              disabled={sourceText.trim().length === 0}
              onClick={() => clearIO()}
            >
                ✕
            </button>
          </div>
          <div className={classes.target}>
            {isLoading ? (
              <Loading height='10em' spinnerSize='6em' />
            ) : (<p>{translationText}</p>)}
          </div>
        </div>
      </div>
    </section>
  )
}

export default Translator;