// Data Structur for a Modepicker in the LocalStorage
export interface ModeLocalStorage {
    walk: {
        enabled: boolean,
        search_profile: {
            profile: string,
            max_duration: number
        }
    },
    bike: {
        enabled: boolean,
        max_duration: number
    },
    car: {
        enabled: boolean,
        max_duration: number,
        use_parking: boolean
    }
}


// Safe a value in LocalStorage
export const setLocalStorage = (key: string, value: any) => {
    try {
        localStorage.setItem(key, JSON.stringify(value));   
    } catch (ex) {}
}


// Get a value from LocalStorage
export const getFromLocalStorage = (key: string) => {
    try {
        return JSON.parse(localStorage.getItem(key) as string);
    } catch (ex) {}
}