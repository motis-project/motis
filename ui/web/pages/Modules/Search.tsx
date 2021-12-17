import React from 'react';

import { Modepicker } from './ModePicker';
import { DatePicker } from './DatePicker';

export const Search: React.FC = () => {
    
    return (
        <div id='search'>
            <div className='pure-g gutters'>
                <div className='pure-u-1 pure-u-sm-12-24 from-location'>
                    <div>
                        <form>
                            <div className='label'>
                                Start
                            </div>
                            <div className='gb-input-group'>
                                <div className='gb-input-icon'>
                                    <i className='icon'>place</i>
                                    </div>
                            <input className='gb-input' tabIndex={1} /></div>
                        </form>
                        <div className='paper hide'>
                            <ul className='proposals'></ul>
                        </div>
                    </div>
                    <Modepicker />
                    <div className='swap-locations-btn'>
                        <label className='gb-button gb-button-small gb-button-circle gb-button-outline gb-button-PRIMARY_COLOR disable-select'>
                            <input type='checkbox' />
                            <i className='icon'>swap_vert</i>
                        </label>
                    </div>
                </div>
                <div className='pure-u-1 pure-u-sm-12-24'>
                    <DatePicker />
                </div>
            </div>
            <div className='pure-g gutters'>
                <div className='pure-u-1 pure-u-sm-12-24 to-location'>
                    <div>
                        <div>
                            <div className='label'>Ziel</div>
                            <div className='gb-input-group'>
                                <div className='gb-input-icon'><i className='icon'>place</i></div>
                                <input className='gb-input' tabIndex={2} />
                            </div>
                        </div>
                        <div className='paper hide'>
                            <ul className='proposals'></ul>
                        </div>
                    </div>
                    <Modepicker />
                </div> 
                <div className='pure-u-1 pure-u-sm-9-24'>
                    <div>
                        <div className='label'>Uhrzeit</div>
                        <div className='gb-input-group'>
                            <div className='gb-input-icon'><i className='icon'>schedule</i></div><input
                                className='gb-input' tabIndex={4} />
                            <div className='gb-input-widget'>
                                <div className='hour-buttons'>
                                    <div><a
                                            className='gb-button gb-button-small gb-button-circle gb-button-outline gb-button-PRIMARY_COLOR disable-select'><i
                                                className='icon'>chevron_left</i></a></div>
                                    <div><a
                                            className='gb-button gb-button-small gb-button-circle gb-button-outline gb-button-PRIMARY_COLOR disable-select'><i
                                                className='icon'>chevron_right</i></a></div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                <div className='pure-u-1 pure-u-sm-3-24 time-option'>
                    <div>
                        <label>
                            Abfahrt
                        <input type='radio' id='search-forward' name='time-option' />
                        </label>
                    </div>
                    <div>
                        <label>
                            Ankunft
                        <input type='radio' id='search-backward' name='time-option' />
                        </label>
                    </div>
                </div>
            </div>
        </div>
    )
}