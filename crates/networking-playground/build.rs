fn main() {
    if std::env::var("CARGO_CFG_TARGET_ARCH").as_deref() == Ok("xtensa") {
        println!("cargo:rustc-link-arg-bins=-Tlinkall.x");
    }
}
