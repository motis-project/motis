import React, { useState } from 'react';

import equal from 'deep-equal';

import { Translations } from '../App/Localization';
import { Mode } from '../Types/IntermodalRoutingTypes';
import { getFromLocalStorage, ModeLocalStorage, setLocalStorage } from '../App/LocalStorage';


export const Modepicker: React.FC<{'translation': Translations, 'title': String, 'modes': Mode[], 'setModes': React.Dispatch<React.SetStateAction<Mode[]>>, 'localStorageModes': string}> = (props) => {
    
    // Foot
    // Boolean used to track if the Foot Mode is selected
    const [footSelected, setFootSelected] = useState<boolean>(false);

    // Current value of the Foot Mode Slider
    const [footMaxDurationSlider, setFootMaxDurationSlider] = useState<number>(30);

    // Complete Mode Object which will be returned upon closing the Modepicker Component if not undefined
    const [footMode, setFootMode] = useState<Mode>(undefined);
    
    // Selected Value in the Profile Dropdown Menu in Foot Mode
    const [profilePicker, setProfilePicker] = useState<String>('default');

    // Bike
    // Boolean used to track if the Bike Mode is selected
    const [bikeSelected, setBikeSelected] = useState<boolean>(false);

    // Current value of the Bike Mode Slider
    const [bikeMaxDurationSlider, setBikeMaxDurationSlider] = useState<number>(30);

    // Complete Mode Object which will be returned upon closing the Modepicker Component if not undefined
    const [bikeMode, setBikeMode] = useState<Mode>(undefined);
    
    // Car
    // Boolean used to track if the Car Mode is selected
    const [carSelected, setCarSelected] = useState<boolean>(false);

    // Current value of the Car Mode Slider
    const [carMaxDurationSlider, setCarMaxDurationSlider] = useState<number>(30);

    // Complete Mode Object which will be returned upon closing the Modepicker Component if not undefined
    const [carMode, setCarMode] = useState<Mode>(undefined);

    // Boolean used to track if Parking in the Car Mode is selected
    const [useParking, setUseParking] = useState<boolean>(false);

    
    // Boolean used to track if the ModePicker is being displayed or not
    const [modePickerVisible, setModePickerVisible] = useState<boolean>(false);

    // Initial load of the Component from LocalStorage
    React.useEffect(() => {
        let modes: ModeLocalStorage = getFromLocalStorage(props.localStorageModes);
        // If LocalStorage is empty, initialize it
        if (modes !== null) {
            setFootSelected(modes.walk.enabled);
            setFootMaxDurationSlider(modes.walk.search_profile.max_duration);
            setProfilePicker(modes.walk.search_profile.profile);
            setBikeSelected(modes.bike.enabled);
            setBikeMaxDurationSlider(modes.bike.max_duration);
            setCarSelected(modes.car.enabled);
            setCarMaxDurationSlider(modes.car.max_duration);
            setUseParking(modes.car.use_parking);
        } else {
            setLocalStorage(props.localStorageModes, {walk: {enabled: false, search_profile: {profile: 'default', max_duration: 30}}, bike: {enabled: false, max_duration: 30}, car: {enabled: false, max_duration: 30, use_parking: false}});
        };
        // set ppr options for the map
        window.portEvents.sub('mapInitFinished', () => {
            window.portEvents.pub('setPPRSearchOptions', {'duration_limit': footMaxDurationSlider*60, 'profile': profilePicker});
        });
    }, []);

    // Update return Value for Foot Mode if any part of this Mode is changed
    React.useEffect(() => {
        if (footSelected){
            setFootMode({ mode_type: 'FootPPR', mode: { search_options: { profile: profilePicker, duration_limit: footMaxDurationSlider * 60 } }});
            //set ppr options for the map
            window.portEvents.pub('setPPRSearchOptions', {'duration_limit': footMaxDurationSlider*60, 'profile': profilePicker});
        }
    }, [footMaxDurationSlider, footSelected, profilePicker]);
    
    // Update return Value for Bike Mode if any part of this Mode is changed
    React.useEffect(() => {
        if (bikeSelected) {
            setBikeMode({ mode_type: 'Bike', mode: { max_duration: bikeMaxDurationSlider * 60 } });
        }
    },[bikeMaxDurationSlider, bikeSelected]);

    // Update return Value for Car Mode if any part of this Mode is changed
    React.useEffect(() => {
        if (carSelected) {
            if (useParking) {
                setCarMode({ mode_type: 'CarParking', mode: { max_car_duration: carMaxDurationSlider * 60, ppr_search_options: { profile: 'default', duration_limit: 300 } } });
            } else {
                setCarMode({ mode_type: 'Car', mode: { max_duration: carMaxDurationSlider * 60 } });
            }
        }
    }, [carMaxDurationSlider, carSelected, useParking]);

    // Check which Modes are selected and should be returned
    const getModeArr = () => {
        let res = []

        if (footSelected) {
            res = [footMode];
        }

        if (bikeSelected) {
            res = [...res, bikeMode]
        }
        
        if (carSelected) {
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
                                let modes = getModeArr();
                                // If the mode objects in the datepicker differ from the ones displayed and used in Search.tsx, then update those when closing the modepicker
                                if (!equal(modes, props.modes)) {
                                    props.setModes(modes);
                                    setLocalStorage(props.localStorageModes, {walk: {enabled: footSelected, search_profile: {profile: profilePicker, max_duration: footMaxDurationSlider}}, bike: {enabled: bikeSelected, max_duration: bikeMaxDurationSlider}, car: {enabled: carSelected, max_duration: carMaxDurationSlider, use_parking: useParking}});
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
                                        checked={footSelected} 
                                        onChange={() => {
                                            setFootSelected(!footSelected)
                                        }}/>
                                {props.translation.connections.walk}
                            </label>
                        </legend>
                        <div className='option'>
                            <div className='label'>{props.translation.search.searchProfile.profile}</div>
                            <div className='profile-picker'>
                                <select onChange={(e) => setProfilePicker(e.target.value)} defaultValue={profilePicker.toString()}>
                                    <option value='default'>{props.translation.searchProfiles.default}</option>
                                    <option value='accessibility1'>{props.translation.searchProfiles.accessibility1}</option>
                                    <option value='wheelchair'>{props.translation.searchProfiles.wheelchair}</option>
                                    <option value='elevation'>{props.translation.searchProfiles.elevation}</option>
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
                                        checked={bikeSelected}  
                                        onChange={() => {
                                            setBikeSelected(!bikeSelected);
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
                                        checked={carSelected} 
                                        onChange={() => {
                                            setCarSelected(!carSelected);
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
                                        checked={useParking}
                                        onChange={() => {
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