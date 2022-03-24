import React from 'react';

export const RailvizTooltipStation: React.FC<{'station': any}> = (props) => {
    return (
        <div className='station-name'>{props.station}</div>
    );
}