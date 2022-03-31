import React, { useEffect, useState } from 'react';
import moment from 'moment';

import { EventInfo } from '../Types/Connection';

export const Delay: React.FC<{'event': EventInfo}> = (props) => {
    
    const [delay, setDelay] = useState<number>(props.event.schedule_time - props.event.time);
    const [isNeg, setIsNeg] = useState<boolean>(true);

    useEffect(() => {
        if(delay < 0){
            setIsNeg(true);
        }else{
            setIsNeg(false);
        }
    }, [delay]);

    return (
        <div className={delay === 0 ?
                            'delay'
                        :
                            isNeg ?
                                'delay neg-delay'
                            :
                                'delay pos-delay'}>
            <span>{delay !== 0 ? ((isNeg ? '+' : '-') + moment.unix(delay).format('m')) : ''}</span>
        </div>
    );
}