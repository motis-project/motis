import React from 'react';

import { SearchOptions } from './PPRTypes';

export interface Mode {
    mode_type: string,
    mode: { search_options: SearchOptions}
}


const switchToMode = (mode: string, old_mode: Mode, setModes: React.Dispatch<React.SetStateAction<[Mode]>>) => {

    setModes([{mode_type: mode, mode: old_mode.mode}])
}


export const Modepicker: React.FC<{'modes': [Mode], 'setModes': React.Dispatch<React.SetStateAction<[Mode]>>}> = (props) => {
    return (
        <div className='mode-picker'>
            <div className='mode-picker-btn'>
                            <div className={props.modes[0].mode_type == 'FootPPR' ? 'mode enabled' : 'mode'} onClick={() => switchToMode('FootPPR', props.modes[0], props.setModes)}><i className='icon'>directions_walk</i></div>
                            <div className={props.modes[0].mode_type == 'BikePPR' ? 'mode enabled' : 'mode'} onClick={() => switchToMode('BikePPR', props.modes[0], props.setModes)}><i className='icon'>directions_bike</i></div>
                            <div className={props.modes[0].mode_type == 'CarPPR' ? 'mode enabled' : 'mode'} onClick={() => switchToMode('CarPPR', props.modes[0], props.setModes)}><i className='icon'>directions_car</i></div>
                        </div>
                        <div className='mode-picker-editor'>
                            <div className='header'>
                                <div className='sub-overlay-close'><i className='icon'>close</i></div>
                                <div className='title'>Verkehrsmittel am Ziel</div>
                            </div>
                            <div className='content'>
                                <fieldset className='mode walk'>
                                    <legend className='mode-header'><label><input type='checkbox' />Fußweg</label>
                                    </legend>
                                    <div className='option'>
                                        <div className='label'>Profil</div>
                                        <div className='profile-picker'><select>
                                                <option value='default'>Standard</option>
                                                <option value='accessibility1'>Auch nach leichten Wegen
                                                    suchen</option>
                                                <option value='wheelchair'>Rollstuhl</option>
                                                <option value='elevation'>Weniger Steigung</option>
                                            </select></div>
                                    </div>
                                    <div className='option'>
                                        <div className='label'>Maximale Dauer (Minuten)</div>
                                        <div className='numeric slider control'><input type='range' min='0'
                                                max='30' step='1' /><input type='text' /></div>
                                    </div>
                                </fieldset>
                                <fieldset className='mode bike disabled'>
                                    <legend className='mode-header'><label><input
                                                type='checkbox' />Fahrrad</label></legend>
                                    <div className='option'>
                                        <div className='label'>Maximale Dauer (Minuten)</div>
                                        <div className='numeric slider control'><input type='range' min='0'
                                                max='30' step='1' /><input type='text' /></div>
                                    </div>
                                </fieldset>
                                <fieldset className='mode car disabled'>
                                    <legend className='mode-header'><label><input type='checkbox' />Auto</label>
                                    </legend>
                                    <div className='option'>
                                        <div className='label'>Maximale Dauer (Minuten)</div>
                                        <div className='numeric slider control'><input type='range' min='0'
                                                max='30' step='1' /><input type='text' /></div>
                                    </div>
                                    <div className='option'><label><input type='checkbox' />Parkplätze
                                            verwenden</label></div>
                                </fieldset>
                            </div>
                        </div>
                    </div>
    )
}