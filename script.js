document.getElementById("mobile-nav-icon").addEventListener(
    "click", function(_event) {
        let target = document.getElementById("mobile-menu");
        if (target.style.display === "block") {
            target.style.display = "none";
        } else {
            target.style.display = "block";
        }
    }
)