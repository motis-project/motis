import React from 'react';

export const RailvizTooltipTrain: React.FC<{'train': any}> = (props) => {
    return (
        <>
            <div className='transport-name'>{props.train.name}</div>
            <div className='departure'>
                <span className='station'>{props.train.departureStation}</span>
                <div className='time no-delay-infos'>{/** muss in abhaengikeit von train gesetzt werden */}
                    <span className='schedule'>{props.train.departureTime}</span>
                </div>
            </div>
            <div className='arrival'>
            <span className='station'>{props.train.arrivalStation}</span>
                <div className='time no-delay-infos'>{/** muss in abhaengikeit von train gesetzt werden */}
                    <span className='schedule'>{props.train.arrivalTime}</span>
                </div>
            </div>
        </>
    );
}