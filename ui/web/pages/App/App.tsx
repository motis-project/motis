import React from 'react';

import { OverlayView } from '../Views/OverlayView';

export const Main: React.FC = () => {
    return(
        <div className='app'>
            {//<MapView />
            }
            <OverlayView />
            {//<StationSearchView />
            }
            {//<SimTimePickerView />
            }
        </div>
    );
};