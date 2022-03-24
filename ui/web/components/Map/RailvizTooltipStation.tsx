import React, { useEffect, useState } from 'react';

export const RailvizTooltipStation: React.FC<{'station': any}> = (props) => {

    const [station, setStation] = useState<any>(props.station);

    useEffect(() => {
        if(props.station){
            setStation(props.station);
        }
    }, [props.station]);

    return (
        <div className='station-name'>{station}</div>
    );
}