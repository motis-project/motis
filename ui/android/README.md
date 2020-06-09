# Regenerate Java Flatbuffers Classes

    find protocol -name '*.fbs' -exec ./build/rel/deps/flatbuffers/flatc32 --java -I protocol -o ui/android/app/src/main/java/ {} \;


Only commit classes from modules which are used in the app!
