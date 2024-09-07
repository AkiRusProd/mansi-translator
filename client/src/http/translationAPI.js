// eslint-disable-next-line
import { guestInstance, authInstance } from './axios'

export const translate = async (data) => {
    try {
        const response = await guestInstance.post('translate', data)
        return response.data
    } catch (e) {
        alert(e.response.data.message)
        return false
    }
}