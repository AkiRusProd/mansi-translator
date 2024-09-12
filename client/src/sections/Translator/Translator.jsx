// eslint-disable-next-line
import React, { useState, useEffect, useRef} from 'react';
import classes from './Translator.module.css';
import { translate } from '../../http/translationAPI';
import Loading from '../Loading';
import RatingModal from './RatingModal';
import ImproveModal from './ImproveModal';

const addSymb = {
  'lower': [
    'а̄', 'ē', 'ё̄', 'ӣ', 'ӈ', 'о̄', 'ӯ', 'ы̄', 'э̄', 'ю̄', 'я̄'
  ],
  'upper': [
    'А̄', 'Ē', 'Ё̄', 'Ӣ', 'Ӈ', 'О̄', 'Ӯ', 'Ы̄', 'Э̄', 'Ю̄', 'Я̄'
  ]
}

const textLayout = {
  'rus': {
    'rus': 'Русский',
    'mansi': 'Мансийский',
    'rate_trans': 'Оценить перевод',
    'improve_trans': 'Улучшить перевод',
    'text_input': 'Введите текст',
    'trans_here': 'Здесь будет перевод',
    'rate_trans_long': 'Оцените качество перевода',
    'improve_trans_long': 'Если перевод оказался некорректен, дайте нам знать',
    'close': 'Закрыть',
    'send': 'Отправить',
    'switch_page_language': 'Нēлм пēнтуӈкв',
    'improve_disabled': 'Введите текст для перевода',
    'thankyou-rate': 'Спасибо за оценку!',
    'thankyou-improve': 'Спасибо за обратную связь!',
    'source_text': 'Исходный текст',
    'your_translation': 'Ваш перевод',
    'enter_your_translation': 'Введите свой перевод'

  },
  'mansi': {
    'rus': 'Русь',
    'mansi': 'Мāньси',
    'rate_trans': 'Ла̄тыӈ толмащлан ва̄рмаль янытлаӈкв',
    'improve_trans': 'Ла̄тыӈ ю̄нтуӈкв',
    'text_input': 'Потырлтэ̄н',
    'trans_here': 'Тыт тах толмащлаӈкв паты',
    'rate_trans_long': 'Ла̄тыӈ толмащлан ва̄рмалит янытлым о̄ньселы̄н',
    'improve_trans_long': 'Ла̄тыӈ толмащлым ёмасыг ке о̄лы, ма̄навн ла̄ве̄н',
    'close': 'Пӯмасаӈкв',
    'send': 'Кēтуӈкве',
    'switch_page_language': 'Сменить язык',
    'improve_disabled': 'Введите текст для переводаа̄а̄а̄',
    'thankyou-rate': 'Спасибо за оценку!а̄а̄а̄а̄',
    'thankyou-improve': 'Спасибо за обратную связь!а̄а̄а̄',
    'source_text': 'Исходный текста̄а̄а̄',
    'your_translation': 'Ваш перевода̄а̄а̄',
    'enter_your_translation': 'Введите свой перевода̄а̄а̄а̄'
  }
}

const ThankYouModal = (props) => (
  <div className={classes["modal-thanks"]}>
    <span onClick={() => props.isOpened(false)}>✕</span>
    <p>{textLayout[props.pageLanguage]['thankyou-improve']}</p>
  </div>
);


const Translator = () => {
  const [isFeedbackModalOpen, setIsFeedbackModalOpen] = useState(false)
  const [isThankYouModalOpen, setIsThankYouModalOpen] = useState(false)
  const [sourceLng, setSourceLng] = useState('rus')
  const [targetLng, setTargetLng] = useState('mansi')
  const [translationText, setTranslationText] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [sourceText, setSourceText] = useState('')
  const [symbType, setSymbType] = useState('lower')
  const [pageLanguage, setPageLanguage] = useState('rus')
  const [isRating, setIsRating] = useState(false)
  const typingTimeoutRef = useRef(null)

  const switchLanguages = () => {
    let src = sourceLng.slice()
    let target = targetLng.slice()
    setSourceLng(target)
    setTargetLng(src)
    if (translationText.length > 0) {
      setIsLoading(true)
      let src_text = translationText.slice()
      setSourceText(src_text)
      sendTranslation(src_text, target, src)
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
    else if (e.key === 'Shift') {
      setSymbType('upper')
    }
  }

  const handleKeyUp = (e) => {
    if (e.key === 'Shift') {
      setSymbType('lower')
    }
  }

  const sendTranslation = async (text, source_lng = sourceLng, target_lng = targetLng) => {
    try {
      // await sleep(200)
      if (text.trim().length === 0) {
        clearIO()
        return
      }
      const data = {
        text: text.trim(),
        source_lang: (source_lng === 'mansi') ? ('mansi_Cyrl') : ('rus_Cyrl'),
        target_lang: (target_lng === 'mansi') ? ('mansi_Cyrl') : ('rus_Cyrl')
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
      <section id={classes["translator"]}>
        {isRating ? (
          <RatingModal 
            pageLanguage={pageLanguage}
            textLayout={textLayout}
            setIsRating={setIsRating} 
            sourceText={sourceText}
            translationText={translationText}
            sourceLng={sourceLng}
            targetLng={targetLng}
          />
          ) : ('')}
        {
          isFeedbackModalOpen ? (
            <ImproveModal 
            pageLanguage={pageLanguage}
            textLayout={textLayout}
            setIsRating={setIsRating} 
            sourceText={sourceText}
            translationText={translationText}
            sourceLng={sourceLng}
            targetLng={targetLng}
            setThanks={setIsThankYouModalOpen}
            isOpened={setIsFeedbackModalOpen}
            />
        ) : ('')
        }
        {
          isThankYouModalOpen ? (
            <ThankYouModal 
              isOpened={setIsThankYouModalOpen}
              pageLanguage={pageLanguage}
            />
          ) : ('')
        }
        <div className={classes['swith-page-language']}>
          <button
            onClick={() => setPageLanguage(pageLanguage === 'rus' ? ('mansi') : 'rus')}
          >
            {textLayout[pageLanguage]['switch_page_language']}
          </button>
        </div>
        <div class={classes["container"]}>
          <div class={classes["language-toggle"]}>
              <button 
                id={classes["russian"]}
                class={classes["active"]}
              >
                {textLayout[pageLanguage][sourceLng]}
              </button>
              <button id={classes["switch"]} onClick={() => switchLanguages()}>⇄</button>
              <button 
                id={classes["mansi"]}
              >
                {textLayout[pageLanguage][targetLng]}
              </button>
          </div>

          <div class={classes["translate-area"]}>
              <div class={classes["input-output"]}>
                  <label id={classes["input-label"]}>{textLayout[pageLanguage][sourceLng]}</label>
                  <textarea
                    onChange={handleInputChange}
                    onKeyDown={handleKeyPress}
                    onKeyUp={handleKeyUp}
                    placeholder={textLayout[pageLanguage]['text_input']}
                    value={sourceText}
                  />
                  <div 
                    class={classes["letter-buttons"]}
                    id={classes["mansi-letters"]}
                    style={{visibility: sourceLng === 'mansi' ? ('visible') : ('hidden')}}
                  >
                      {addSymb[symbType].map((s, index) => (
                        <button key={index}>{s}</button>
                      ))}
                  </div>
              </div>
              <div class={classes["divider"]}>
              <button 
                disabled={sourceText.trim().length === 0}
                onClick={() => clearIO()}
              >
                  ✕
              </button>
              </div>
              <div class={classes["input-output"]}>
                  <label id={classes["output-label"]}>{textLayout[pageLanguage][targetLng]}</label>
                  {isLoading ? (
                    <Loading height='6em' spinnerSize='5em'/>
                  ) : (
                  <textarea 
                    readOnly={true}
                    placeholder={textLayout[pageLanguage]['trans_here']}
                    value={translationText}>
                    
                  </textarea>
                  )}
                  
              </div>
          </div>

          <div class={classes["translate-buttons"]}>
              <div class={classes["left"]}>
                  {/* <button id={classes["translate"]}>Перевести</button>
                  <button id={classes["swap-languages"]}>Поменять языки</button> */}
              </div>
              <div class={classes["right"]}>
                  <button id={classes["rate"]} onClick={() => setIsRating(true)}>
                    {textLayout[pageLanguage]['rate_trans']}
                  </button>
                  <button 
                    id={classes["improve"]}
                    onClick={() => setIsFeedbackModalOpen(true)}
                    disabled={!sourceText.trim() || isLoading}
                    title={!sourceText.trim() || isLoading ? (textLayout[pageLanguage]['improve_disabled']) : ('')}
                  >
                    {textLayout[pageLanguage]['improve_trans']}
                  </button>
              </div>
          </div>
      </div>

  </section>
  )
}

export default Translator;