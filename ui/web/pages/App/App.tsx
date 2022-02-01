import React from 'react';

import Maybe, { nothing } from 'true-myth/maybe';
import { useRouter} from 'next/router';

import { Overlay } from '../Modules/Overlay';
import { Translations, deTranslations, enTranslations } from '../Modules/Localization';
import { StationSearch } from '../Modules/StationSearch';
import { MapContainer } from '../Modules/MapContainer';

let visible = false;


const getQuery = (): Translations => {
    let router = useRouter();
    let { locale } = router.query;
    if (locale === 'en') {
        return enTranslations
    }
    return deTranslations
}


export const App: React.FC = () => {

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
    
    return(
        <div className='app'>
            {/* visible && <MapView />*/}
            <MapContainer />
            <Overlay translation={getQuery()}/>
            {//<StationSearchView />
            }
            <StationSearch />
        </div>
    );
};