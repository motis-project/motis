import React from 'react';

import { useRouter} from 'next/router';

import { Overlay } from '../Modules/Overlay';
import { Translations, deTranslations, enTranslations, plTranslations } from '../Modules/Localization';
import { StationSearch } from '../Modules/StationSearch';
import { MapContainer } from '../Modules/MapContainer';


<<<<<<< HEAD
const getQuery = (): Translations => {
    let router = useRouter();
    let { locale } = router.query;
    if (locale === 'de') {
        return deTranslations;
    } else if (locale === 'pl') {
        return plTranslations;
    }
    return enTranslations;
}
=======
>>>>>>> 43f7aee6fc9ae754740c5ab796110e850a8f40bd
declare global{
    interface Window {
        portEvents : any;
    }
}  

<<<<<<< HEAD

export const App: React.FC = () => {
=======
export const Main: React.FC = () => {
    
    /*let Model: Boolean = true;

    const [showMap, setShowMap] = React.useState(false);

    // Switch between Dark and Light Mode
    const [darkMode, setDarkMode] = React.useState(true);

    React.useEffect(() =>{
        console.log("Hello World")
        visible = !visible
    }, [showMap])

    React.useEffect(() =>{
        Model = darkMode
    }, [darkMode])*/
>>>>>>> 43f7aee6fc9ae754740c5ab796110e850a8f40bd
    
    return (
        <div className='app'>
            {/* visible && <MapView />*/}
            <MapContainer translation={getQuery()}/>
            <Overlay translation={getQuery()}/>
            {//<StationSearchView />
            }
            <StationSearch />
        </div>
    );
};