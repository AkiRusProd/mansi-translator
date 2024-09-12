// eslint-disable-next-line
import { guestInstance, authInstance } from './axios'

export const rateTranslation = async (data) => {
    try {
        await guestInstance.post('rate', data)
        return true
    } catch (e) {
        alert("Ошибка. Повторите запрос позже")
        return false
    }
}

export const improveTranslation = async (data) => {
    try {
        await guestInstance.post('improve', data)
        return true
    } catch (e) {
        alert("Ошибка. Повторите запрос позже")
        return false
    }
}
