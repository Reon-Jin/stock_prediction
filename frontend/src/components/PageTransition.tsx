import { motion } from "framer-motion";
import { PropsWithChildren } from "react";

export function PageTransition({ children }: PropsWithChildren) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 18 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -18 }}
      transition={{ duration: 0.28, ease: [0.22, 1, 0.36, 1] }}
      className="page-transition"
    >
      {children}
    </motion.div>
  );
}
