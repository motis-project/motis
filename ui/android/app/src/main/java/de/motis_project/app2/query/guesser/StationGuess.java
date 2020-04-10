package de.motis_project.app2.query.guesser;

public class StationGuess implements  Comparable<StationGuess> {
    static final int SERVER_GUESS = 0;
    static final int FAVORITE_GUESS = 1;

    final String eva;
    final String name;
    final int priority;
    final int type;

    public StationGuess(String eva, String name, int count, int type) {
        this.eva = eva;
        this.name = name;
        this.priority = count;
        this.type = type;
    }

    @Override
    public String toString() {
        return name;
    }

    @Override
    public boolean equals(Object obj) {
        if (obj instanceof StationGuess) {
            StationGuess other = (StationGuess) obj;
            return eva.equals(other.eva);
        }
        return false;
    }

    @Override
    public int hashCode() {
        return eva.hashCode();
    }

    @Override
    public int compareTo(StationGuess o) {
        return Integer.compare(o.priority, priority);
    }
}