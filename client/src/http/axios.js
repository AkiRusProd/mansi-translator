import axios from 'axios'

const serverUrl = 'http://127.0.0.1:8000'

const guestInstance = axios.create({
    baseURL: serverUrl 
})

const authInstance = axios.create({
    baseURL: serverUrl
})


const authInterceptor = (config) => {
    const token = localStorage.getItem('token')
    if (token) {
        config.headers.authorization = 'Bearer ' + localStorage.getItem('token')
    }
    return config
}
authInstance.interceptors.request.use(authInterceptor)

export {
    guestInstance,
    authInstance
}