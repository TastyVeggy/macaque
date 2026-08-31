proc glob_recursive {dir} {
    set res {}
    foreach f [glob -nocomplain -directory $dir *] {
        if {[file isdirectory $f]} {
            lappend res {*}[glob_recursive $f]
        } elseif {[file extension $f] == ".sv" || [file extension $f] == ".v"} {
            lappend res $f
        }
    }
    return $res
}
