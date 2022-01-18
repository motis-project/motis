import { createStore, Store } from 'vuex'
import TripResponseContent from '../models/TripResponseContent'
import StationGuess from '../models/StationGuess'
import AddressGuess from '../models/AddressGuess'

export default createStore({
  state: {
    connections: [] as TripResponseContent[],
    startInput: {} as StationGuess | AddressGuess,
    destinationInput: {} as StationGuess | AddressGuess
  },
})


declare module '@vue/runtime-core' {
  interface State {
    connections: TripResponseContent[],
    startInput: StationGuess | AddressGuess,
    destinationInput: StationGuess | AddressGuess
  }

  interface ComponentCustomProperties {
    $store: Store<State>
  }
}
