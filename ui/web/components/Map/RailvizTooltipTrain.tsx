import React, { useEffect, useState } from 'react';
import moment from 'moment';

export const RailvizTooltipTrain: React.FC<{'train': any}> = (props) => {

    const [train, setTrain] = useState<any>(props.train || {names: [''], departureTime: 0, departureStation: '', arrivalTime: 0, arrivalStation: ''});

    useEffect(() => {
        if(props.train){
            setTrain(props.train);
        }
    }, [props.train]);

    return (
        <>
            <div className='transport-name'>{train.names[0]}</div>
            <div className='departure'>
                <span className='station'>{train.departureStation}</span>
                <div className='time no-delay-infos'>{/** muss in abhaengikeit von train gesetzt werden */}
                    <span className='schedule'>{moment(train.departureTime).format('HH:mm')}</span>
                </div>
            </div>
            <div className='arrival'>
            <span className='station'>{train.arrivalStation}</span>
                <div className='time no-delay-infos'>{/** muss in abhaengikeit von train gesetzt werden */}
                    <span className='schedule'>{moment(train.arrivalTime).format('HH:mm')}</span>
                </div>
            </div>
        </>
    );
}