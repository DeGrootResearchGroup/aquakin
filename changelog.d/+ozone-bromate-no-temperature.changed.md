`ozone_bromate` no longer declares a temperature condition `T`: none of its
rates depend on temperature, so asking for one suggested a temperature
correction the model does not have. Code that still passes `T` keeps working;
the value is ignored, as it always was.
