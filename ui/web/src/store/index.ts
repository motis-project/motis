import { createStore, Store } from 'vuex'
import TripResponseContent from '../models/TripResponseContent'
import StationGuess from '../models/StationGuess'
import AddressGuess from '../models/AddressGuess'
import Position from '../models/SmallTypes/Position'
import { RVConnection } from '../services/MOTISMapService'

export default createStore({
  state: {
    connections: [] as TripResponseContent[],
    startInput: {} as StationGuess | AddressGuess,
    destinationInput: {} as StationGuess | AddressGuess,
    // eslint-disable-next-line @typescript-eslint/no-empty-function
    setStart: () => {},
    // eslint-disable-next-line @typescript-eslint/no-empty-function
    setDestination: () => {},
    mapConnections: [] as RVConnection[]
  },
})


declare module '@vue/runtime-core' {
  interface State {
    connections: TripResponseContent[],
    startInput: StationGuess | AddressGuess,
    destinationInput: StationGuess | AddressGuess
    areConnectionsDropped: boolean,
    setStart: (pos: Position) => void,
    setDestination: (pos: Position) => void,
    mapConnections: RVConnection[]
  }

  interface ComponentCustomProperties {
    $store: Store<State>
  }
}
