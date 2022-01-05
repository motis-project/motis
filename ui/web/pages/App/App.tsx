import React from 'react';

import Maybe, { nothing } from 'true-myth/maybe';

import { Overlay } from '../Modules/Overlay';
import { SimTimeOverlay } from '../Modules/SimTimeOverlay';

let visible = false;


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
    
    return(
        <div className='app'>
            {/* visible && <MapView />*/}
            <Overlay />
            {//<StationSearchView />
            }
            <SimTimeOverlay />
        </div>
    );
};