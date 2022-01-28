import React, { useState } from 'react';
import { Translations } from './Localization';
import { Mode } from './IntermodalRoutingTypes';


export const Modepicker: React.FC<{'translation': Translations, 'title': String, 'modes': Mode[], 'setModes': React.Dispatch<React.SetStateAction<Mode[]>>}> = (props) => {
    
    // Foot
    // FootSelected
    const [footSelected, setFootSelected] = useState<boolean>(false);

    // FootMaxDurationSlider
    const [footMaxDurationSlider, setFootMaxDurationSlider] = useState<number>(30);

    // Foot Mode
    const [footMode, setFootMode] = useState<Mode>(undefined);
    
        // Profile-Picker Value
        const [profilePicker, setProfilePicker] = useState<String>('default');

    // Bike
    // BikeSelected
    const [bikeSelected, setBikeSelected] = useState<boolean>(false);

    // BikeMaxDurationSlider
    const [bikeMaxDurationSlider, setBikeMaxDurationSlider] = useState<number>(30);

    // Bike Mode
    const [bikeMode, setBikeMode] = useState<Mode>(undefined);
    
    // Car
    // CarSelected
    const [carSelected, setCarSelected] = useState<boolean>(false);

    // CarMaxDurationSlider
    const [carMaxDurationSlider, setCarMaxDurationSlider] = useState<number>(30);

    // Car Mode
    const [carMode, setCarMode] = useState<Mode>(undefined);

    // use CarParking
    const [useParking, setUseParking] = useState<boolean>(false);

    
    // Mode-Picker-Visible
    const [modePickerVisible, setModePickerVisible] = useState<boolean>(false);

    // Fetch new Intermodal Data
    const [newFetch, setNewFetch] = useState<boolean>(false);

    React.useEffect(() => {
        props.modes.forEach((mode: any) => {
            if (mode.mode_type === 'FootPPR') {
                setFootMode(mode);
                setFootMaxDurationSlider(mode.mode.search_options.duration_limit / 60);
                setFootSelected(true);
                setProfilePicker(mode.mode.search_options.profile);
            } else if (mode.mode_type === 'Bike') {
                setBikeMode(mode);
                setBikeMaxDurationSlider(mode.mode.max_duration / 60);
                setBikeSelected(true);
            } else {
                if (mode.mode_type === 'CarParking') {
                    setUseParking(true);
                    setCarMaxDurationSlider(mode.mode.max_car_duration / 60);
                }else {
                    setCarMaxDurationSlider(mode.mode.max_duration / 60);
                }
                setCarMode(mode);
                setCarSelected(true);
            }
        })
    }, [])

    React.useEffect(() => {
        if (footSelected){
            setFootMode({ mode_type: 'FootPPR', mode: { search_options: { profile: profilePicker, duration_limit: footMaxDurationSlider * 60 } }})
            setNewFetch(true);
        } else {
            setFootMode(undefined);
        }
    }, [footMaxDurationSlider, footSelected, profilePicker]);
    
    React.useEffect(() => {
        if (bikeSelected) {
            setBikeMode({ mode_type: 'Bike', mode: { max_duration: bikeMaxDurationSlider * 60 } });
            setNewFetch(true);
        } else {
            setBikeMode(undefined);
        }
    },[bikeMaxDurationSlider, bikeSelected]);

    React.useEffect(() => {
        if (carSelected) {
            if (useParking) {
                setCarMode({ mode_type: 'CarParking', mode: { max_car_duration: carMaxDurationSlider * 60, ppr_search_options: { profile: 'default', duration_limit: 300 } } });
            } else {
                setCarMode({ mode_type: 'Car', mode: { max_duration: carMaxDurationSlider * 60 } });
            }
            setNewFetch(true);
        } else {
            setCarMode(undefined);
        }
    }, [carMaxDurationSlider, carSelected, useParking]);

    const getModeArr = () => {
        let res = []

        if (footMode !== undefined) {
            res = [footMode];
        }

        if (bikeMode !== undefined) {
            res = [...res, bikeMode]
        }
        
        if (carMode !== undefined) {
            res = [...res, carMode]
        }
        return res;
    }

    return (
        <div className='mode-picker'>
            <div className='mode-picker-btn' onClick={() => setModePickerVisible(true)}>
                <div className={footSelected ? 'mode enabled' : 'mode'}><i className='icon'>directions_walk</i></div>
                <div className={bikeSelected ? 'mode enabled' : 'mode'}><i className='icon'>directions_bike</i></div>
                <div className={carSelected ? 'mode enabled' : 'mode'}><i className='icon'>directions_car</i></div>
            </div>
            <div className='mode-picker-editor' style={modePickerVisible ? {display: 'flex'} : {display: 'none'}}>
                <div className='header'>
                    <div    className='sub-overlay-close' 
                            onClick={() => {
                                if (newFetch) {
                                    props.setModes(getModeArr());
                                    setNewFetch(false);
                                }
                                setModePickerVisible(false);
                                }}>
                        <i className='icon'>close</i>
                    </div>
                    <div className='title'>{props.title}</div>
                </div>
                <div className='content'>
                    <fieldset className='mode walk'>
                        <legend className='mode-header'>
                            <label>
                                <input  type='checkbox' 
                                        defaultChecked={footSelected} 
                                        onClick={() => {
                                            setFootSelected(!footSelected)
                                            setNewFetch(true);
                                        }}/>
                                {props.translation.connections.walk}
                            </label>
                        </legend>
                        <div className='option'>
                            <div className='label'>{props.translation.search.searchProfile.profile}</div>
                            <div className='profile-picker'>
                                <select onChange={(e) => setProfilePicker(e.target.value)}>
                                    <option value='default' selected={profilePicker === 'default'} >{props.translation.searchProfiles.default}</option>
                                    <option value='accessibility1' selected={profilePicker === 'accessibility1'}>{props.translation.searchProfiles.accessibility1}</option>
                                    <option value='wheelchair' selected={profilePicker === 'wheelchair'}>{props.translation.searchProfiles.wheelchair}</option>
                                    <option value='elevation' selected={profilePicker === 'elevation'}>{props.translation.searchProfiles.elevation}</option>
                                </select></div>
                        </div>
                        <div className='option'>
                            <div className='label'>{props.translation.search.maxDuration}</div>
                            <div className='numeric slider control'>
                                <input  type='range' 
                                        min='0'
                                        max='30' 
                                        step='1' 
                                        value={footMaxDurationSlider} 
                                        onChange={(e) => setFootMaxDurationSlider(e.currentTarget.valueAsNumber)} />
                                <input  type='text' 
                                        value={footMaxDurationSlider} 
                                        onChange={(e) => setFootMaxDurationSlider(e.currentTarget.valueAsNumber > 30 ? 30 : e.currentTarget.valueAsNumber)} />
                            </div>
                        </div>
                    </fieldset>
                    <fieldset className='mode bike'>
                        <legend className='mode-header'>
                            <label>
                                <input  type='checkbox' 
                                        defaultChecked={bikeSelected}  
                                        onClick={() => {
                                            setBikeSelected(!bikeSelected);
                                            setNewFetch(true);
                                        }}/>
                                {props.translation.connections.bike}
                            </label>
                        </legend>
                        <div className='option'>
                            <div className='label'>{props.translation.search.maxDuration}</div>
                            <div className='numeric slider control' ><input type='range' min='0'
                                    max='30' step='1' value={bikeMaxDurationSlider} onChange={(e) => setBikeMaxDurationSlider(e.currentTarget.valueAsNumber)} /><input type='text' value={bikeMaxDurationSlider} onChange={(e) => setBikeMaxDurationSlider(e.currentTarget.valueAsNumber > 30 ? 30 : e.currentTarget.valueAsNumber)} /></div>
                        </div>
                    </fieldset>
                    <fieldset className='mode car'>
                        <legend className='mode-header'>
                            <label>
                                <input  type='checkbox' 
                                        defaultChecked={carSelected} 
                                        onClick={() => {
                                            setCarSelected(!carSelected);
                                            setNewFetch(true);
                                        }}/>
                                {props.translation.connections.car}
                            </label>
                        </legend>
                        <div className='option'>
                            <div className='label'>{props.translation.search.maxDuration}</div>
                            <div className='numeric slider control'><input type='range' min='0'
                                    max='30' step='1' value={carMaxDurationSlider} onChange={(e) => setCarMaxDurationSlider(e.currentTarget.valueAsNumber)} /><input type='text' value={carMaxDurationSlider} onChange={(e) => setCarMaxDurationSlider(e.currentTarget.valueAsNumber > 30 ? 30 : e.currentTarget.valueAsNumber)} /></div>
                        </div>
                        <div className='option'>
                            <label>
                                <input  type='checkbox' 
                                        defaultChecked={useParking}
                                        onClick={() => {
                                            setUseParking(!useParking);
                                        }}/>
                                {props.translation.search.useParking}
                            </label>
                        </div>
                    </fieldset>
                </div>
            </div>
        </div>
    )
}