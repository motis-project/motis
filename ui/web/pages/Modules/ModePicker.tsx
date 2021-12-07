import React from "react";

export const Modepicker: React.FC = () => {
    return (
        <div className="mode-picker">
            <div className="mode-picker-btn">
                            <div className="mode enabled"><i className="icon">directions_walk</i></div>
                            <div className="mode"><i className="icon">directions_bike</i></div>
                            <div className="mode"><i className="icon">directions_car</i></div>
                        </div>
                        <div className="mode-picker-editor">
                            <div className="header">
                                <div className="sub-overlay-close"><i className="icon">close</i></div>
                                <div className="title">Verkehrsmittel am Ziel</div>
                            </div>
                            <div className="content">
                                <fieldset className="mode walk">
                                    <legend className="mode-header"><label><input type="checkbox" />Fußweg</label>
                                    </legend>
                                    <div className="option">
                                        <div className="label">Profil</div>
                                        <div className="profile-picker"><select>
                                                <option value="default">Standard</option>
                                                <option value="accessibility1">Auch nach leichten Wegen
                                                    suchen</option>
                                                <option value="wheelchair">Rollstuhl</option>
                                                <option value="elevation">Weniger Steigung</option>
                                            </select></div>
                                    </div>
                                    <div className="option">
                                        <div className="label">Maximale Dauer (Minuten)</div>
                                        <div className="numeric slider control"><input type="range" min="0"
                                                max="30" step="1" /><input type="text" /></div>
                                    </div>
                                </fieldset>
                                <fieldset className="mode bike disabled">
                                    <legend className="mode-header"><label><input
                                                type="checkbox" />Fahrrad</label></legend>
                                    <div className="option">
                                        <div className="label">Maximale Dauer (Minuten)</div>
                                        <div className="numeric slider control"><input type="range" min="0"
                                                max="30" step="1" /><input type="text" /></div>
                                    </div>
                                </fieldset>
                                <fieldset className="mode car disabled">
                                    <legend className="mode-header"><label><input type="checkbox" />Auto</label>
                                    </legend>
                                    <div className="option">
                                        <div className="label">Maximale Dauer (Minuten)</div>
                                        <div className="numeric slider control"><input type="range" min="0"
                                                max="30" step="1" /><input type="text" /></div>
                                    </div>
                                    <div className="option"><label><input type="checkbox" />Parkplätze
                                            verwenden</label></div>
                                </fieldset>
                            </div>
                        </div>
                    </div>
    )
}